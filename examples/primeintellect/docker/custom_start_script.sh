#!/bin/bash

# Set up SSH key from environment variable
if [ ! -z "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# Set SSH port from environment variable
if [ ! -z "$SSH_PORT" ]; then
    sed -i '/^#*Port /d' /etc/ssh/sshd_config
    echo "Port $SSH_PORT" >> /etc/ssh/sshd_config
fi

echo "Starting SSH server on port ${SSH_PORT:-22}"

# Start SSH service in foreground
exec /usr/sbin/sshd -D