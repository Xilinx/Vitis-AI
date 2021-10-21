# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

override CPPFLAGS += -I../..
override CXXFLAGS += -std=c++11 -Wall -O3 -fPIC -fsanitize=address
override LDFLAGS += $(CXXFLAGS)

all: $(addprefix bin/, simd_test)

obj/targets.o: targets.cc
	@mkdir -p obj
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) targets.cc -o obj/targets.o

obj/simd_test.o: simd_test.cc
	@mkdir -p obj
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) simd_test.cc -o obj/simd_test.o

.DELETE_ON_ERROR:
deps.mk: $(wildcard *.cc) $(wildcard *.h) Makefile
	set -eu; for file in *.cc; do \
		target=obj/$${file##*/}; target=$${target%.*}.o; \
		$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -MM -MT \
		"$$target" "$$file"; \
	done | sed -e ':b' -e 's-../[^./]*/--' -e 'tb' >$@
-include deps.mk

bin/simd_test: $(addprefix obj/, targets.o simd_test.o)
	@mkdir -p bin
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	[ ! -d obj ] || $(RM) -r -- obj/
	[ ! -d bin ] || $(RM) -r -- bin/
	[ ! -d lib ] || $(RM) -r -- lib/

.PHONY: clean all
