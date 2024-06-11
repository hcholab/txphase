# Copyright (C) 2023 Gramine contributors
# SPDX-License-Identifier: BSD-3-Clause

ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

SELF_EXE = eagle 

.PHONY: all
all: $(SELF_EXE) eagle.manifest
ifeq ($(SGX),1)
all: eagle.manifest.sgx eagle.sig
endif

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

eagle.manifest: eagle.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dself_exe=$(SELF_EXE) \
		$< $@

# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# see the phasing example for details on this workaround.
eagle.manifest.sgx eagle.sig: sgx_sign
	@:

.INTERMEDIATE: sgx_sign
sgx_sign: eagle.manifest $(SELF_EXE)
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
	$(GRAMINE) eagle 

.PHONY: clean
clean:
	$(RM) -rf *.token *.sig *.manifest.sgx *.manifest 
