APP = libxperfmon

PYTHON_VERSION = 38
SRC=$(wildcard src/*.cpp)
OBJ = $(SRC:.cpp=.o)

NAME := xperfmon
MAJOR = 0.1
MINOR = 0
VERSION = $(MAJOR).$(MINOR)
INCLUDE = inc/

all: lib$(NAME).so

lib$(NAME).so.$(VERSION): $(OBJ)
	$(CXX) $(LDFLAGS_BOOST) $(LDFLAGS) $(OBJ) -shared -Wl,-soname,lib$(NAME).so.$(MAJOR) -lboost_python$(PYTHON_VERSION) -o lib$(NAME).so.$(VERSION)

lib$(NAME).so: lib$(NAME).so.$(VERSION)
	rm -f lib$(NAME).so.$(MAJOR) lib$(NAME).so
	ln -s lib$(NAME).so.$(VERSION) lib$(NAME).so.$(MAJOR)
	ln -s lib$(NAME).so.$(MAJOR) lib$(NAME).so

%.o: %.cpp
	$(CXX) $(CFLAGS) -I$(INCLUDE) -c -fPIC $(SRC) -o $@

clean:
	rm -f *.so
