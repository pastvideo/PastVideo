use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use pastvideo::{Embedder, GeminiConfig, GeminiEmbedder, RemoteConfig, RemoteEmbedder};

fn serve_once(response_body: String) -> (String, thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut bytes = vec![];
        let mut chunk = [0_u8; 4096];
        loop {
            let read = stream.read(&mut chunk).unwrap();
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&chunk[..read]);
            let Some(header_end) = find_subslice(&bytes, b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|value| value.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            if bytes.len() >= header_end + 4 + content_length {
                break;
            }
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream.write_all(response.as_bytes()).unwrap();
        String::from_utf8(bytes).unwrap()
    });
    (format!("http://{address}"), handle)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[test]
fn gemini_adapter_sends_retrieval_instruction_and_dimensions() {
    let values = vec!["0.125"; 128].join(",");
    let (base_url, request) = serve_once(format!(r#"{{"embedding":{{"values":[{values}]}}}}"#));
    let embedder = GeminiEmbedder::new(GeminiConfig {
        api_key: "test-key".into(),
        base_url,
        model: "gemini-embedding-2".into(),
        dimensions: 128,
        timeout: Duration::from_secs(5),
    })
    .unwrap();

    let embedding = embedder.embed_text("a red bicycle").unwrap();
    assert_eq!(embedding.len(), 128);
    let request = request.join().unwrap();
    assert!(request.contains("x-goog-api-key: test-key"));
    assert!(request.contains("task: search result | query: a red bicycle"));
    assert!(request.contains("\"outputDimensionality\":128"));
}

#[test]
fn remote_adapter_sends_documented_contract_and_bearer_token() {
    let (base_url, request) = serve_once(r#"{"data":[{"embedding":[1.0,0.0,0.5]}]}"#.into());
    let embedder = RemoteEmbedder::new(RemoteConfig {
        endpoint: format!("{base_url}/embed"),
        api_key: "remote-token".into(),
        model: "clip-service".into(),
        dimensions: 3,
        timeout: Duration::from_secs(5),
    })
    .unwrap();

    assert_eq!(embedder.embed_text("traffic at night").unwrap().len(), 3);
    let request = request.join().unwrap();
    assert!(request.contains("authorization: Bearer remote-token"));
    assert!(request.contains("\"kind\":\"text\""));
    assert!(request.contains("\"model\":\"clip-service\""));
    assert!(request.contains("\"dimensions\":3"));
}
