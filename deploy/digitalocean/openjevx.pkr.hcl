# DigitalOcean Marketplace 1-Click image: Ubuntu 24.04 + the OpenJevX server and 8-bit model as a service.
#   export DIGITALOCEAN_TOKEN=...   (sec run DIGITALOCEAN_TOKEN -- packer build .)
#   packer init . && packer build .
# The snapshot lands in the account's images; submit it in the Vendor Portal (docs/DEPLOY.md).
packer {
  required_plugins {
    digitalocean = {
      source  = "github.com/digitalocean/digitalocean"
      version = ">= 1.4.0"
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

source "digitalocean" "openjevx" {
  api_token     = env("DIGITALOCEAN_TOKEN")
  image         = "ubuntu-24-04-x64"
  region        = "nyc3"
  size          = "s-2vcpu-4gb"
  ssh_username  = "root"
  snapshot_name = "openjevx-${local.version}-${formatdate("YYYYMMDDhhmm", timestamp())}"
}

build {
  sources = ["source.digitalocean.openjevx"]

  provisioner "shell" {
    inline = [
      "cloud-init status --wait",
      "export DEBIAN_FRONTEND=noninteractive",
      "apt-get update -q && apt-get -y -q -o Dpkg::Options::=--force-confold upgrade",
      "apt-get install -y -q curl ca-certificates ufw",
    ]
  }

  provisioner "shell" {
    environment_vars = [
      "OPENJEVX_VERSION=${local.version}",
      "OPENJEVX_BASE=${var.openjevx_base}",
      "OPENJEVX_NO_START=1",
    ]
    script = "../install.sh"
  }

  # DigitalOcean's required cleanup and image check (Marketplace partner rules).
  provisioner "shell" {
    scripts = ["files/90-cleanup.sh", "files/99-img-check.sh"]
  }
}
