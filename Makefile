help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  python     to generate the protobuf python bindings"
	@echo "  clean      to remove generated files and downloaded tools"

PROTO_SOURCEDIR = proto_src
PROTO_SRCS = $(shell find $(PROTO_SOURCEDIR) -name '*.proto')
PYTHON_OUT = py_src
PROTO_PYS  = $(addprefix $(PYTHON_OUT)/,$(PROTO_SRCS:proto_src/%.proto=%_pb2.py))
BUILD_INFO_PY = $(PYTHON_OUT)/mlpiot/proto/_build_info.py

clean:
	rm -f *.pyc */*.pyc */*/*.pyc */*/*/*.pyc
	rm -f $(BUILD_INFO_PY)
	rm -f $(PROTO_PYS)
	rm -rf tools

tools/protoc/bin/protoc:
	mkdir -p tools/protoc/
	curl -L -o tools/protoc.zip https://github.com/google/protobuf/releases/download/v$(PROTOC_VERSION)/protoc-$(PROTOC_VERSION)-$(OS)-$(ARCH).zip
	unzip tools/protoc.zip -d tools/protoc/
	rm tools/protoc.zip

$(PROTO_PYS): $(PROTO_SRCS) tools/protoc/bin/protoc
	tools/protoc/bin/protoc --proto_path=$(PROTO_SOURCEDIR) --python_out=$(PYTHON_OUT) $(PROTO_SRCS)

define BUILD_INFO
PARENT_GIT_COMMIT = '$(GIT_COMMIT)'
PARENT_GIT_TAG = '$(GIT_TAG)'
BUILD_TIME = '$(BUILD_TIME)'
endef
export BUILD_INFO
$(BUILD_INFO_PY):
	@echo "$$BUILD_INFO" > $(BUILD_INFO_PY)

python: $(PROTO_PYS) $(BUILD_INFO_PY)

.PHONY: help clean python

## Commons Vars ##########################################################
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
	OS ?= linux
	GOOS ?= linux
endif
ifeq ($(UNAME_S), Darwin)
	OS ?= osx
	GOOS ?= darwin
endif
ARCH := $(shell uname -m)
ifeq ($(ARCH), unknown)
	GOARCH ?= amd64
	ARCH := x86_64
endif
ifeq ($(ARCH), x86_64)
	GOARCH ?= amd64
endif
ifeq ($(ARCH), i386)
	ARCH = x86_32
	GOARCH ?= 386
endif

PROTOC_VERSION ?=3.11.2
GIT ?= git
GIT_COMMIT := $(shell $(GIT) rev-parse HEAD)
GIT_TAG ?= $(shell $(GIT) describe --tags ${COMMIT} 2> /dev/null || $(GIT) rev-parse --short HEAD)
BUILD_TIME := $(shell LANG=en_US date +"%F_%T_%z")
