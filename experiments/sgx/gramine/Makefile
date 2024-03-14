# Copyright (C) 2023 Gramine contributors
# SPDX-License-Identifier: BSD-3-Clause

ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)
PHASING_DIR = ../../../phasing

SELF_EXE = phasing
ORIG_SELF_EXE = $(PHASING_DIR)/target/release/phasing

.PHONY: all
all: $(SELF_EXE) phasing.manifest
ifeq ($(SGX),1)
all: phasing.manifest.sgx phasing.sig
endif

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

-include $(ORIG_SELF_EXE).d # See also: .cargo/config.toml
$(ORIG_SELF_EXE): $(PHASING_DIR)/Cargo.toml
	cd $(PHASING_DIR)
	RUSTFLAGS="$(RUSTFLAGS)" cargo +nightly build $(PROFILE)

$(SELF_EXE): $(ORIG_SELF_EXE)
	cp $< $@ 

phasing.manifest: $(PHASING_DIR)/phasing.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dself_exe=$(SELF_EXE) \
		$< $@

# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# see the phasing example for details on this workaround.
phasing.manifest.sgx phasing.sig: sgx_sign
	@:

.INTERMEDIATE: sgx_sign
sgx_sign: phasing.manifest $(SELF_EXE)
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

ifeq ($(SGX),)
GRAMINE = gramine-direct
else
GRAMINE = gramine-sgx
endif

.PHONY: start-gramine
start-gramine: all
	$(GRAMINE) phasing

.PHONY: clean
clean:
	$(RM) -rf *.token *.sig *.manifest.sgx *.manifest result-* OUTPUT

.PHONY: distclean
distclean: clean
	$(RM) -rf target/ Cargo.lock
