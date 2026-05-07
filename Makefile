##
# mri_fusion
#
# @file
# @version 0.1

# variables
build_dir = build
exec = MRI_Fusion

# default target: build and run
all: build run

# configure and compile
build:
	@mkdir -p $(build_dir)
	@cmake -S . -B $(build_dir) -DCMAKE_BUILD_TYPE=Release
	@cmake --build $(build_dir)

# run the executable
# run:
# 	@./$(build_dir)/$(exec)
run:
	@export XDG_RUNTIME_DIR=/run/user/$$(id -u) && \
	export WAYLAND_DISPLAY=$$(ls /run/user/$$(id -u)/wayland-* 2>/dev/null | head -n 1 | xargs basename || echo "") && \
	if [ -n "$$WAYLAND_DISPLAY" ]; then \
		export QT_QPA_PLATFORM=wayland; \
	else \
		export QT_QPA_PLATFORM=xcb; \
	fi && \
	./$(build_dir)/$(exec)

# clean build artifacts
clean:
	@rm -rf $(build_dir)
	@echo "build directory cleaned."

# help menu
help:
	@echo "usage:"
	@echo "  make        - build and run the project"
	@echo "  make build  - just compile"
	@echo "  make run    - just run the existing executable"
	@echo "  make clean  - remove all build files"

.phony: all build run clean help



# end
