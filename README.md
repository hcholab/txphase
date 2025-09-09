# TX-Phase

TX-Phase is a secure haplotype phasing framework based on the Trusted Execution Environment (TEE) technology, implemented for Intel SGX using [the Gramine framework](https://gramineproject.io/). 

The algorithm implemented in this package is described in the following publication:
> **TX-Phase: Secure Phasing of Private Genomes in a Trusted Execution Environment** \
> Natnatee Dokmai, Kaiyuan Zhu, S. Cenk Sahinalp, Hyunghoon Cho \
> Genome Research, 2025

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
- [Rust Nightly](https://www.rust-lang.org/tools/install) (tested with version 1.91.0-nightly)
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
  ```

## Download Test Data and Run TX-Phase
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

### Service provider side
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

## Verify the Results
To verify the switch error rate (SER) of the phased output on the client side, we will use trio-based phasing as the ground truth as follows:

- Install [`bcftools`](https://samtools.github.io/bcftools/bcftools.html):
  ```bash
  sudo apt install bcftools
  ```

- Download the VCF files for the parents (HG003 and HG004) on Chromosome 20, along with the pedigree (`.ped`) file describing the familial relationship between HG002, HG003, and HG004. The full dataset can be downloaded from [this link](https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/AshkenazimTrio/). The pedigree file was created following the [standard format](https://gatk.broadinstitute.org/hc/en-us/articles/360035531972-PED-Pedigree-format).
  ```bash
  wget https://github.com/hcholab/txphase-test-data/raw/refs/heads/main/HG003_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz
  wget https://github.com/hcholab/txphase-test-data/raw/refs/heads/main/HG004_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz
  wget https://github.com/hcholab/txphase-test-data/raw/refs/heads/main/trio.ped
  ```

- Index the phased output (assuming the filename is `phased.vcf.gz`) and the parents’ VCF files:
  ```bash
  bcftools index phased.vcf.gz
  bcftools index HG003_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz
  bcftools index HG004_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz
  ```

- Merge the trio VCF files into a single VCF file:
  ```bash
  bcftools merge -m none -Oz -o merged.vcf.gz phased.vcf.gz HG003_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz HG004_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz
  ```

- Use the `bcftools` plugin `trio-switch-rate` to compute the SER:
  ```bash
  bcftools +trio-switch-rate merged.vcf.gz -- -p trio.ped
  ```
  The output should include lines similar to the following, indicating that the SER is 1.29%:
  ```text
  # TRIO	[2]Father	[3]Mother	[4]Child	[5]nTested	[6]nMendelian Errors	[7]nSwitch	[8]nSwitch (%)
  TRIO	HG003	HG004	HG002	8424	1	109	1.29
  ```


## Contact Information
Ko Dokmai, natnatee@cmkl.ac.th \
Hoon Cho, hoon.cho@yale.edu
