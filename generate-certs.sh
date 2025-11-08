#!/bin/bash
# Generate TLS certificates for Flower Federated Learning

set -e

# Configuration
CERT_DIR="./certs"
DAYS_VALID=365
COUNTRY="US"
STATE="California"
CITY="San Francisco"
ORG="Flower Federated Learning"
CN_CA="Flower CA"
CN_SERVER="flower-server"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Generating TLS Certificates for Flower ===${NC}"

# Create certificate directory
mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

# 1. Generate CA (Certificate Authority)
echo -e "\n${YELLOW}Step 1: Generating Certificate Authority (CA)...${NC}"
openssl genrsa -out ca.key 4096

openssl req -new -x509 -days $DAYS_VALID -key ca.key -out ca.crt \
    -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/CN=$CN_CA"

echo -e "${GREEN}✓ CA certificate generated: ca.crt, ca.key${NC}"

# 2. Generate Server Certificate
echo -e "\n${YELLOW}Step 2: Generating Server Certificate...${NC}"
openssl genrsa -out server.key 4096

openssl req -new -key server.key -out server.csr \
    -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/CN=$CN_SERVER"

# Create config file for SAN (Subject Alternative Names)
cat > server.ext <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = flower-server
DNS.3 = *.compute.amazonaws.com
DNS.4 = *.compute-1.amazonaws.com
DNS.5 = *.compute.internal
DNS.6 = *.googleapis.com
IP.1 = 127.0.0.1
EOF

# Add server public IP placeholder
if [ ! -z "$SERVER_PUBLIC_IP" ]; then
    echo "IP.2 = $SERVER_PUBLIC_IP" >> server.ext
fi

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
    -CAcreateserial -out server.crt -days $DAYS_VALID \
    -extfile server.ext

echo -e "${GREEN}✓ Server certificate generated: server.crt, server.key${NC}"

# 3. Generate Client Certificates (optional, for mutual TLS)
echo -e "\n${YELLOW}Step 3: Generating Client Certificates...${NC}"
for i in {1..5}; do
    openssl genrsa -out client$i.key 4096
    
    openssl req -new -key client$i.key -out client$i.csr \
        -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/CN=flower-client-$i"
    
    openssl x509 -req -in client$i.csr -CA ca.crt -CAkey ca.key \
        -CAcreateserial -out client$i.crt -days $DAYS_VALID
    
    echo -e "${GREEN}✓ Client $i certificate generated${NC}"
done

# 4. Verify certificates
echo -e "\n${YELLOW}Step 4: Verifying certificates...${NC}"
openssl verify -CAfile ca.crt server.crt
openssl verify -CAfile ca.crt client1.crt

# 5. Set proper permissions
chmod 600 *.key
chmod 644 *.crt

# 6. Create summary
echo -e "\n${GREEN}=== Certificate Generation Complete ===${NC}"
echo -e "
Certificate files created in: $(pwd)

${YELLOW}Certificate Authority:${NC}
  - ca.crt (public)
  - ca.key (private - keep secure!)

${YELLOW}Server Certificates:${NC}
  - server.crt (public)
  - server.key (private - keep secure!)

${YELLOW}Client Certificates:${NC}
  - client1.crt, client1.key
  - client2.crt, client2.key
  - client3.crt, client3.key
  - client4.crt, client4.key
  - client5.crt, client5.key

${YELLOW}Usage:${NC}
1. Copy ca.crt to all clients
2. Copy server.crt and server.key to server
3. Copy client*.crt and client*.key to respective clients

${YELLOW}For Docker deployment:${NC}
  docker run -v ./certs:/app/certs:ro ...

${YELLOW}Important:${NC}
  - Keep private keys (.key files) secure
  - Never commit private keys to version control
  - Regenerate certificates before they expire
  - Update SERVER_PUBLIC_IP in server.ext if needed
"

# Create .gitignore
cat > .gitignore <<EOF
*.key
*.csr
*.srl
EOF

echo -e "${GREEN}✓ .gitignore created to protect private keys${NC}"

cd ..
echo -e "\n${GREEN}All done! Certificates are ready to use.${NC}"