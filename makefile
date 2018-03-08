all:
	@g++ DECAT.cpp -o DECAT -O2 -std=c++14
test:
	@./DECAT Elm.csv Node.csv Prop.csv
