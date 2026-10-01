# AWS Marketplace AMI: Ubuntu 24.04 + the OpenJevX server and 8-bit model as a service.
#   AWS credentials from the environment (sec run AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY -- packer build .)
#   packer init . && packer build .
# Then add the AMI in the AWS Marketplace Management Portal (docs/DEPLOY.md).
packer {
  required_plugins {
    amazon = {
      source  = "github.com/hashicorp/amazon"
      version = ">= 1.3.0"
    }
  }
}

variable "openjevx_version" {
  type        = string
  default     = ""
  description = "Release to install; empty means deploy/VERSION."
}

locals {
  version = var.openjevx_version != "" ? var.openjevx_version : trimspace(file("${path.root}/../VERSION"))
}

variable "openjevx_base" {
  type        = string
  default     = ""
  description = "Where the release files are downloaded from; empty means the GitHub release."
}

variable "region" {
  type    = string
  default = "us-east-1"
}

source "amazon-ebs" "openjevx" {
  region        = var.region
  instance_type = "t3.medium"
  ssh_username  = "ubuntu"
  ami_name      = "openjevx-${local.version}-${formatdate("YYYYMMDDhhmm", timestamp())}"
  imds_support  = "v2.0"
  source_ami_filter {
    filters = {
      name                = "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"
      virtualization-type = "hvm"
    }
    owners      = ["099720109477"]
    most_recent = true
  }
}

build {
  sources = ["source.amazon-ebs.openjevx"]

  provisioner "shell" {
    inline = [
      "cloud-init status --wait",
      "export DEBIAN_FRONTEND=noninteractive",
      "sudo -E apt-get update -q && sudo -E apt-get -y -q upgrade",
      "sudo -E apt-get install -y -q curl ca-certificates ufw",
    ]
  }

  provisioner "shell" {
    environment_vars = [
      "OPENJEVX_VERSION=${local.version}",
      "OPENJEVX_BASE=${var.openjevx_base}",
      "OPENJEVX_NO_START=1",
    ]
    execute_command = "sudo -E bash '{{ .Path }}'"
    script          = "../install.sh"
  }

  # AWS Marketplace rules: no keys, passwords or history left in the AMI.
  provisioner "shell" {
    inline = [
      "sudo rm -f /opt/openjevx/openjevx.json /opt/openjevx/openjevx.json.done /root/openjevx-password",
      "sudo rm -f /home/ubuntu/.ssh/authorized_keys /root/.ssh/authorized_keys",
      "sudo cloud-init clean --logs",
    ]
  }
}
