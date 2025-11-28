docker run --rm \
  --name pi-docker-runner \
  --privileged \
  --ipc=host \
  --shm-size=8G \
  -p 22:22 \
  -e PUBLIC_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIL5cJCLeYxOFSzEllDCxYJV0slWVhEWLS8FpATCiO4cM your_email@example.com" \
  -e SSH_PORT=22 \
  -v ./custom_start_script.sh:/custom_start_script.sh \
  --entrypoint /bin/bash \
  pitestdocker:latest \
  -c "/custom_start_script.sh"