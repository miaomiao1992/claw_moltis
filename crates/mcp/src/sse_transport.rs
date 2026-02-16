//! HTTP/SSE transport for remote MCP servers (Streamable HTTP transport).
//!
//! Uses HTTP POST for JSON-RPC requests and GET for server-initiated SSE events.
//! Supports optional OAuth Bearer token injection and automatic 401 retry.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use {
    anyhow::{Context, Result, bail},
    reqwest::Client,
    secrecy::ExposeSecret,
    tracing::{debug, info, warn},
};

use crate::{
    auth::SharedAuthProvider,
    traits::McpTransport,
    types::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpTransportError},
};

/// HTTP/SSE-based transport for a remote MCP server.
pub struct SseTransport {
    client: Client,
    url: String,
    next_id: AtomicU64,
    /// Optional auth provider for Bearer token injection.
    auth: Option<SharedAuthProvider>,
}

impl SseTransport {
    /// Create a new SSE transport pointing at the given MCP server URL.
    pub fn new(url: &str) -> Result<Arc<Self>> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .context("failed to build HTTP client for SSE transport")?;

        Ok(Arc::new(Self {
            client,
            url: url.to_string(),
            next_id: AtomicU64::new(1),
            auth: None,
        }))
    }

    /// Create a new SSE transport with an OAuth auth provider.
    pub fn with_auth(url: &str, auth: SharedAuthProvider) -> Result<Arc<Self>> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .context("failed to build HTTP client for SSE transport")?;

        Ok(Arc::new(Self {
            client,
            url: url.to_string(),
            next_id: AtomicU64::new(1),
            auth: Some(auth),
        }))
    }

    /// Build a request builder with optional Bearer token.
    async fn build_post(&self) -> Result<reqwest::RequestBuilder> {
        let mut req = self
            .client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");

        if let Some(token) = match &self.auth {
            Some(auth) => auth.access_token().await?,
            None => None,
        } {
            req = req.header("Authorization", format!("Bearer {}", token.expose_secret()));
        }

        Ok(req)
    }

    /// Send a POST request and handle 401 with auth retry.
    /// Returns the HTTP response or a typed `McpTransportError`.
    async fn send_with_auth_retry(
        &self,
        method: &str,
        body: &impl serde::Serialize,
    ) -> Result<reqwest::Response> {
        // First attempt
        let req = self.build_post().await?;
        let http_resp = req
            .json(body)
            .send()
            .await
            .with_context(|| format!("SSE POST to '{}' for '{method}' failed", self.url))?;

        if http_resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            if let Some(auth) = &self.auth {
                let www_auth = http_resp
                    .headers()
                    .get("www-authenticate")
                    .and_then(|v| v.to_str().ok())
                    .map(String::from);

                info!(method = %method, url = %self.url, "received 401, attempting OAuth re-auth");

                if auth.handle_unauthorized(www_auth.as_deref()).await? {
                    // Retry with new token
                    let req = self.build_post().await?;
                    let retry_resp = req.json(body).send().await.with_context(|| {
                        format!("SSE POST retry to '{}' for '{method}' failed", self.url)
                    })?;

                    if retry_resp.status() == reqwest::StatusCode::UNAUTHORIZED {
                        return Err(McpTransportError::Unauthorized {
                            www_authenticate: retry_resp
                                .headers()
                                .get("www-authenticate")
                                .and_then(|v| v.to_str().ok())
                                .map(String::from),
                        }
                        .into());
                    }

                    return Ok(retry_resp);
                }
            }

            // No auth provider or auth failed
            let www_auth = http_resp
                .headers()
                .get("www-authenticate")
                .and_then(|v| v.to_str().ok())
                .map(String::from);
            return Err(McpTransportError::Unauthorized {
                www_authenticate: www_auth,
            }
            .into());
        }

        Ok(http_resp)
    }
}

#[async_trait::async_trait]
impl McpTransport for SseTransport {
    async fn request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<JsonRpcResponse> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let req = JsonRpcRequest::new(id, method, params);

        debug!(method = %method, id = %id, url = %self.url, "SSE client -> server");

        let http_resp = self.send_with_auth_retry(method, &req).await?;

        if !http_resp.status().is_success() {
            let status = http_resp.status();
            let body = http_resp.text().await.unwrap_or_default();
            bail!("MCP SSE server returned HTTP {status} for '{method}': {body}",);
        }

        let resp: JsonRpcResponse = http_resp
            .json()
            .await
            .with_context(|| format!("failed to parse JSON-RPC response for '{method}'"))?;

        if let Some(ref err) = resp.error {
            bail!(
                "MCP SSE error on '{method}': code={} message={}",
                err.code,
                err.message
            );
        }

        Ok(resp)
    }

    async fn notify(&self, method: &str, params: Option<serde_json::Value>) -> Result<()> {
        let notif = JsonRpcNotification {
            jsonrpc: "2.0".into(),
            method: method.into(),
            params,
        };

        debug!(method = %method, url = %self.url, "SSE client -> server (notification)");

        let http_resp = self.send_with_auth_retry(method, &notif).await?;

        if !http_resp.status().is_success() {
            let status = http_resp.status();
            warn!(method = %method, %status, "SSE notification returned non-success");
        }

        Ok(())
    }

    async fn is_alive(&self) -> bool {
        // Try a lightweight HEAD request to check connectivity.
        let mut req = self
            .client
            .head(&self.url)
            .timeout(std::time::Duration::from_secs(5));

        // Include auth header in health checks too
        if let Some(token) = match &self.auth {
            Some(auth) => auth.access_token().await.ok().flatten(),
            None => None,
        } {
            req = req.header("Authorization", format!("Bearer {}", token.expose_secret()));
        }

        req.send().await.is_ok()
    }

    async fn kill(&self) {
        // For SSE transport, there is no persistent connection to kill.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_transport_creation() {
        let transport = SseTransport::new("http://localhost:8080/mcp");
        assert!(transport.is_ok());
    }

    #[test]
    fn test_sse_transport_invalid_url_still_creates() {
        // reqwest doesn't validate URLs at build time, only at request time
        let transport = SseTransport::new("not-a-url");
        assert!(transport.is_ok());
    }

    #[tokio::test]
    async fn test_sse_transport_is_alive_unreachable() {
        let transport = SseTransport::new("http://127.0.0.1:1/mcp").unwrap();
        assert!(!transport.is_alive().await);
    }

    #[tokio::test]
    async fn test_sse_transport_request_unreachable() {
        let transport = SseTransport::new("http://127.0.0.1:1/mcp").unwrap();
        let result = transport.request("test", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_sse_transport_kill() {
        let transport = SseTransport::new("http://localhost:8080/mcp").unwrap();
        transport.kill().await;
        // Should not panic
    }

    #[test]
    fn test_sse_transport_with_auth_creation() {
        let auth: SharedAuthProvider = Arc::new(crate::auth::NoAuthProvider);
        let transport = SseTransport::with_auth("http://localhost:8080/mcp", auth);
        assert!(transport.is_ok());
    }

    #[tokio::test]
    async fn test_sse_transport_401_without_auth_returns_unauthorized() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/")
            .with_status(401)
            .with_header("www-authenticate", r#"Bearer realm="test""#)
            .create_async()
            .await;

        let transport = SseTransport::new(&server.url()).unwrap();
        let result = transport.request("test", None).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.downcast_ref::<McpTransportError>().is_some());
    }

    #[tokio::test]
    async fn test_sse_transport_200_no_auth() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#)
            .create_async()
            .await;

        let transport = SseTransport::new(&server.url()).unwrap();
        let resp = transport.request("test", None).await.unwrap();
        assert!(resp.result.is_some());
    }

    #[tokio::test]
    async fn test_sse_transport_bearer_header_injected() {
        use secrecy::Secret;

        use crate::auth::{McpAuthProvider, McpAuthState};

        /// Test auth provider that always returns a fixed token.
        struct FixedTokenProvider;

        #[async_trait::async_trait]
        impl McpAuthProvider for FixedTokenProvider {
            async fn access_token(&self) -> Result<Option<Secret<String>>> {
                Ok(Some(Secret::new("test-token-123".to_string())))
            }

            async fn handle_unauthorized(&self, _: Option<&str>) -> Result<bool> {
                Ok(false)
            }

            fn auth_state(&self) -> McpAuthState {
                McpAuthState::Authenticated
            }
        }

        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/")
            .match_header("authorization", "Bearer test-token-123")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#)
            .create_async()
            .await;

        let auth: SharedAuthProvider = Arc::new(FixedTokenProvider);
        let transport = SseTransport::with_auth(&server.url(), auth).unwrap();
        let resp = transport.request("test", None).await.unwrap();
        assert!(resp.result.is_some());
        mock.assert_async().await;
    }
}
