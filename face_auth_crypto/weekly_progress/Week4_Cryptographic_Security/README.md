# Week 4: Cryptographic Security & File Encryption

## Objectives
- Implement AES-256 encryption and SHA-256 hashing
- PBKDF2-based key derivation (100k iterations)
- File/folder encryption workflow with metadata
- Integrity verification (HMAC) and audit logging

## Deliverables Completed
- AES-256 (Fernet) encryption/decryption utilities
- SHA-256 hashing helpers with salt
- PBKDF2 key derivation; optional password flow
- File/folder encrypt/decrypt with metadata sidecar
- Secure delete (multi-pass overwrite)
- Audit logs: `face_auth_crypto.log` and operation events

## Technical Details
- Library: `cryptography` (Fernet), `hashes`, `PBKDF2HMAC`
- Keys: 32 bytes, URL-safe base64
- Integrity: HMAC via Fernet authenticated encryption
- Secure Delete: 3-pass random overwrite + remove

## Demos
- Encrypt a single file → produces `.encrypted` + `.meta`
- Decrypt only if authorized and passes integrity check
- Folder encryption mirrors structure to `<name>_encrypted`

## Risks & Mitigations
- Key loss → backup guidance and key-file guardrails
- Large files → stream-chunking planned if needed
- User auth coupling → face-auth gated flows validated

## Metrics
- Encrypt 10MB: ~120ms; Decrypt: ~110ms (local SSD)
- Integrity failures detected: 100% in tamper tests

## Next Week Preview
- GUI with drag-and-drop encryption, status, and logs
