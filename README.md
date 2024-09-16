# TX-Phase: Secure Phasing of Private Genomes in a Trusted Execution Environment

⚠️ **DISCLAIMER**: This project is in development. Do not use it in production. ⚠️

**TX-Phase** is a secure haplotype phasing framework in a Trusted Execution Environment (TEE), implemented in Intel SGX using [the Gramine framework](https://gramineproject.io/). 

The instructions below have been tested with a virtual machine in the [Microsoft Azure DCsv3 series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/general-purpose/dcsv3-series?tabs=sizebasic) and Ubuntu 22.04.

## Installation Requirements
- Ubuntu 22.04
- Build dependencies
  - Build essentials
  ```bash
  sudo apt install build-essential
  ```
  - CMake
  ```bash
  sudo apt install cmake
  ```
- [Rust Nightly](https://www.rust-lang.org/tools/install) (tested with version 1.83.0-nightly)
  ```bash
  # Install rustup by following the on-screen instruction
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
      
  # Install Rust Nightly
  rustup toolchain install nightly
  ```  
- [Gramine](https://gramineproject.io/)
  ```bash
  sudo curl -fsSLo /usr/share/keyrings/gramine-keyring.gpg https://packages.gramineproject.io/gramine-keyring.gpg
  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/gramine-keyring.gpg] https://packages.gramineproject.io/ $(lsb_release -sc) main" \
  | sudo tee /etc/apt/sources.list.d/gramine.list

  sudo curl -fsSLo /usr/share/keyrings/intel-sgx-deb.asc https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key
  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-sgx-deb.asc] https://download.01.org/intel-sgx/sgx_repo/ubuntu $(lsb_release -sc) main" \
  | sudo tee /etc/apt/sources.list.d/intel-sgx.list

  sudo apt-get update
  sudo apt-get install gramine
  ```
- Generate Gramine's SGX signing key
  ```bash
  gramine-sgx-gen-private-key
  ```
  

## Configuration & Build
- Client
  ```bash
  cargo +nightly build --release -p client
  ```
- Service Provider (with SGX)
  ```bash
  cargo +nightly build --release -p host
  cargo +nightly build --release -p phasing
  make SGX=1
  ```
- Service Provider (testing without SGX)
  ```bash
  cargo +nightly build --release -p host
  cargo +nightly build --release -p phasing
  make
  ```

## Download and run the test data
### Client side
- Download the GIAB target sample HG002 (Chr 20)
  ```bash
  wget https://github.com/hcholab/txphase-test-data/raw/main/HG002_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz
  ```

   The rest of the dataset can be downloaded from [this link](https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/AshkenazimTrio/).

  
- Start the client
  ```bash
  target/release/client --input HG002_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz --output phased.vcf.gz &
  ```

  To specify the Service Provider's IP address:
  ```bash
  target/release/client --sp-ip-address <SP_IP_ADDRESS> --input HG002_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz --output phased.vcf.gz
  ```

### Service Provider side
- Download the 1KG Phase 3 reference panel (Chr 20) in the M3VCF format
  ```bash
  wget https://github.com/hcholab/txphase-test-data/raw/main/20.1000g.Phase3.v5.With.Parameter.Estimates.m3vcf.gz
  ```
  
  The rest of the dataset can be downloaded from [Minimac4's website](https://genome.sph.umich.edu/wiki/Minimac4). Additionally, reference panels in the VCF format can be converted to the M3VCF using [Minimac3](https://genome.sph.umich.edu/wiki/Minimac3_Usage).
   
- Download a genetic map file (Chr 20)
  ```bash
  wget https://github.com/hcholab/txphase-test-data/raw/main/chr20.b37.gmap
  ```

  The rest of the genetic map files can be downloaded from [SHAPEIT4's repository](https://github.com/odelaneau/shapeit4/tree/master/maps).

- Start the phasing service (with SGX)
  ```bash
  target/release/host --ref-panel 20.1000g.Phase3.v5.With.Parameter.Estimates.m3vcf.gz --genetic-map chr20.b37.gmap &
  gramine-sgx phasing/phasing
  ```

- Start the phasing service (testing without SGX)
  ```bash
  target/release/host --ref-panel 20.1000g.Phase3.v5.With.Parameter.Estimates.m3vcf.gz --genetic-map chr20.b37.gmap &
  target/release/phasing
  ```
