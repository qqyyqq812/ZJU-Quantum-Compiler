# Huawei Cloud Deployment

The public website uses the REST API. MCP is optional and should be deployed
only after the REST API is healthy.

```text
GitHub Pages / docs/index.html
  -> https://api.your-domain.com/api/compile
  -> src.server.app:app

MCP client, optional
  -> https://mcp.your-domain.com/mcp
  -> src.server.mcp_app:app
```

## Recommended product

Use Huawei Cloud Flexus or ECS with Docker. This project loads Qiskit, Torch,
and checked-in model weights, so a normal long-running server is simpler than a
serverless function.

Minimum practical configuration:

- OS: Ubuntu 22.04 or 24.04
- CPU/RAM: 2 vCPU / 4 GB minimum, 4 vCPU / 8 GB recommended
- Disk: 40 GB minimum, 80 GB recommended
- Bandwidth: 2 Mbps minimum, 5 Mbps recommended
- Inbound security group:
  - `22/tcp`: your own IP only
  - `80/tcp`: public
  - `443/tcp`: public
  - `8080/tcp`: temporary test only, close after Nginx is working
  - `8081/tcp`: do not expose publicly unless you deliberately publish MCP

For a final GitHub Pages deployment, the API should use HTTPS. If the server is
in mainland China and you bind a domain to it, complete ICP filing first. If
you do not want ICP filing now, choose a non-mainland region such as Hong Kong
and bind a normal HTTPS domain there.

## 1. Install Docker on the server

```bash
sudo apt update
sudo apt install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker
docker --version
```

## 2. Clone and run the REST API

```bash
git clone https://github.com/qqyyqq812/ZJU-Quantum-Compiler.git
cd ZJU-Quantum-Compiler

docker build -f Dockerfile.api -t zju-quantum-api:latest .
docker run -d \
  --name zju-quantum-api \
  --restart unless-stopped \
  -e PORT=8080 \
  -p 127.0.0.1:8080:8080 \
  zju-quantum-api:latest
```

Verify on the server:

```bash
curl http://127.0.0.1:8080/api/status
curl http://127.0.0.1:8080/api/compile \
  -H 'Content-Type: application/json' \
  -d '{"example":"ghz5","topology":"tokyo"}'
```

## 3. Put Nginx in front of the API

```bash
sudo apt install -y nginx
sudo tee /etc/nginx/sites-available/zju-quantum-api >/dev/null <<'NGINX'
server {
    listen 80;
    server_name api.your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/zju-quantum-api /etc/nginx/sites-enabled/zju-quantum-api
sudo nginx -t
sudo systemctl reload nginx
```

Point `api.your-domain.com` to the server public IP, then add HTTPS with your
preferred certificate method. After HTTPS is ready, verify:

```bash
curl https://api.your-domain.com/api/status
```

## 4. Point the website to Huawei Cloud

For a temporary check, open the page with:

```text
https://qqyyqq812.github.io/ZJU-Quantum-Compiler/?api=https://api.your-domain.com
```

For production, replace `PUBLIC_API_BASE` in `docs/index.html` with:

```js
const PUBLIC_API_BASE = "https://api.your-domain.com";
```

Then commit and push that single website change.

## Optional MCP helper

Deploy MCP only after the REST API works:

```bash
docker build -f Dockerfile.mcp -t zju-quantum-mcp:latest .
docker run -d \
  --name zju-quantum-mcp \
  --restart unless-stopped \
  -e PORT=8081 \
  -p 127.0.0.1:8081:8081 \
  zju-quantum-mcp:latest

curl http://127.0.0.1:8081/health
```

Use a separate Nginx server block such as `mcp.your-domain.com` if you need
remote MCP access. The public website does not require this service.

## Updating after a new GitHub push

```bash
cd ZJU-Quantum-Compiler
git pull
docker build -f Dockerfile.api -t zju-quantum-api:latest .
docker stop zju-quantum-api || true
docker rm zju-quantum-api || true
docker run -d \
  --name zju-quantum-api \
  --restart unless-stopped \
  -e PORT=8080 \
  -p 127.0.0.1:8080:8080 \
  zju-quantum-api:latest
```
