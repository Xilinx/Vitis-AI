# #####################################################################
# FSE - Makefile
# Copyright (C) Yann Collet 2015
# GPL v2 License
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# You can contact the author at :
#  - Public forum froup : https://groups.google.com/forum/#!forum/lz4c
# #####################################################################
# This is just a launcher for the Makefile within test directory
# #####################################################################

PROGDIR?= programs

.PHONY: clean test

default: test

all:
	$(MAKE) -C $(PROGDIR) $@

test:
	$(MAKE) -C $(PROGDIR) $@

clean:
	$(MAKE) -C $(PROGDIR) $@

gpptest: clean
	@echo ---- test g++ compilation ----
	$(MAKE) -C $(PROGDIR) all CC=g++ CFLAGS="-O3 -Wall -Wextra -Wundef -Wshadow -Wcast-align -Wcast-qual -Werror"

armtest: clean
	@echo ---- test ARM compilation ----
	CFLAGS="-O3 -Werror" $(MAKE) -C $(PROGDIR) bin CC=arm-linux-gnueabi-gcc

clangtest: clean
	@echo ---- test clang compilation ----
	CFLAGS="-O3 -Werror -Wconversion -Wno-sign-conversion" CC=clang $(MAKE) -C $(PROGDIR) all

clangpptest: clean
	@echo ---- test clang++ compilation ----
	$(MAKE) -C $(PROGDIR) all CC=clang++ CFLAGS="-O3 -Wall -Wextra -Wundef -Wshadow -Wcast-align -Wcast-qual -x c++ -Werror"

staticAnalyze: clean
	@echo ---- static analyzer - scan-build ----
	scan-build --status-bugs -v $(MAKE) -C $(PROGDIR) all CFLAGS=-g   # does not work well; too many false positives

sanitize: clean
	@echo ---- check undefined behavior - sanitize ----
	CC=clang CFLAGS="-g -O3 -fsanitize=undefined" $(MAKE) -C $(PROGDIR) test   FSETEST="-i5000" FSEU16TEST=-i2000
	CC=clang CFLAGS="-g -O3 -fsanitize=undefined" $(MAKE) -C $(PROGDIR) test32 FSETEST="-i5000" FSEU16TEST=-i2000


