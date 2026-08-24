# n8n channel host on AWS — the ONLY always-on infrastructure in the project.
# Everything else is batch (pipeline), static (Vercel) or scale-to-zero.
#
# This replaces deploy/main.tf (GCP), which is kept for reference. The move was
# forced, not chosen: the GCP project's billing is disabled pending KYC, which
# stopped the Compute VM running n8n AND the Cloud Run API in one go. Nothing
# about the stack was GCP-specific — docker-compose.yml and Caddyfile are
# unchanged and copied across as-is.
#
#   cd deploy/aws && terraform init && terraform apply
#
# Earth Engine is unaffected and stays on Google: it is a DATA SOURCE, not a
# hosting choice, and its free non-commercial tier works regardless of the
# project's billing state (verified — S5P queries still return).

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  description = "ap-south-1 (Mumbai): closest region to the cities this serves, and to the team."
  default     = "ap-south-1"
}

variable "instance_type" {
  # 2 vCPU / 1 GiB, and free-tier eligible for 12 months. n8n idles at a few
  # hundred MB; the GCP box it replaces was an e2-small (2 GiB) and was never
  # close to using it.
  default = "t3.micro"
}

variable "ssh_cidr" {
  description = "Who may SSH. Narrow this to your own IP/32 if you can."
  default     = "0.0.0.0/0"
}

# Latest Debian 12, resolved rather than hardcoded: a pinned AMI id is
# region-specific and goes stale, and this is the kind of thing that fails
# months later for no visible reason.
data "aws_ami" "debian" {
  most_recent = true
  owners      = ["136693071363"] # Debian's official account
  filter {
    name   = "name"
    values = ["debian-12-amd64-*"]
  }
}

resource "aws_security_group" "n8n" {
  name        = "n8n-web"
  description = "Caddy: 80 for ACME challenge, 443 for webhooks"

  ingress {
    description = "HTTP: ACME challenge and redirect to 443"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "HTTPS: webhook endpoint for Telegram and the web form"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "n8n-web" }
}

# Without a key pair the instance boots fine and then nobody can get into it —
# and every remaining step (copy the compose stack, start it, read Caddy's logs
# when the cert fails) is an SSH session. Registering the operator's existing
# public key is cheaper than rebuilding the box to add one later.
resource "aws_key_pair" "operator" {
  key_name   = "aircase-operator"
  public_key = file(pathexpand(var.ssh_public_key))
}

variable "ssh_public_key" {
  description = "Local public key registered on the instance."
  default     = "~/.ssh/id_ed25519.pub"
}

resource "aws_instance" "n8n" {
  ami                    = data.aws_ami.debian.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.operator.key_name
  vpc_security_group_ids = [aws_security_group.n8n.id]

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  # Installs Docker ONLY. The compose stack is copied and started by hand
  # (deploy/aws/README.md) — baking app config into a boot script is where these
  # setups fail invisibly, and a four-command SSH session is easier to debug than
  # a cloud-init log. Same reasoning as the GCP version this replaces.
  user_data = <<-EOT
    #!/bin/bash
    set -e
    apt-get update
    apt-get install -y ca-certificates curl
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    usermod -aG docker admin
  EOT

  tags = { Name = "n8n" }
}

# Elastic IP, for the same reason the GCP build used a static address: Telegram
# registers its webhook against a hostname that resolves HERE, and Caddy's
# certificate is issued for it. If the instance is ever replaced, the address —
# and therefore the webhook and the cert — survives.
resource "aws_eip" "n8n" {
  instance = aws_instance.n8n.id
  domain   = "vpc"
  tags     = { Name = "n8n-ip" }
}

output "public_ip" {
  value = aws_eip.n8n.public_ip
}

output "ssh" {
  value = "ssh admin@${aws_eip.n8n.public_ip}"
}

output "next_step" {
  value = "Point aq-intel.duckdns.org at ${aws_eip.n8n.public_ip}, wait for DNS, then follow deploy/aws/README.md."
}
