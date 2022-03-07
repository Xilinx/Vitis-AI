LIBS= -lxperfmon -lpthread
PRJROOT=../../
INCLUDE=$(PRJROOT)/inc
OBJ = basic.o

%.o: %.cpp
	${CXX} -c -o $@ $< -std=c++11 -I$(INCLUDE) -I$(INCLUDE)/arch

apm_app: $(OBJ)
	$(CXX) -o $@ $^ $(LIBS) -L$(PRJROOT)

.PHONY: clean
clean:
	rm -f *.o apm_app
