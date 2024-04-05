# Copyright (C) 2023 Gramine contributors
# SPDX-License-Identifier: BSD-3-Clause

ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

SELF_EXE = target/release/phasing

.PHONY: all
all: $(SELF_EXE) phasing/phasing.manifest
ifeq ($(SGX),1)
all: phasing/phasing.manifest.sgx phasing/phasing.sig
endif

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

# Note that we're compiling in release mode regardless of the DEBUG setting passed
# to Make, as compiling in debug mode results in an order of magnitude's difference in
# performance that makes testing by running a benchmark with ab painful. The primary goal
# of the DEBUG setting is to control Gramine's loglevel.
-include $(SELF_EXE).d # See also: .cargo/config.toml

phasing/phasing.manifest: phasing/phasing.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dself_exe=$(SELF_EXE) \
		$< $@

# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# see the phasing example for details on this workaround.
phasing/phasing.manifest.sgx phasing/phasing.sig: sgx_sign
	@:

.INTERMEDIATE: sgx_sign
sgx_sign: phasing/phasing.manifest $(SELF_EXE)
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

ifeq ($(SGX),)
GRAMINE = gramine-direct
else
GRAMINE = gramine-sgx
endif

.PHONY: run
run: all
	$(GRAMINE) phasing/phasing

.PHONY: clean
clean:
	$(RM) -rf phasing/*.token phasing/*.sig phasing/*.manifest.sgx phasing/*.manifest 
