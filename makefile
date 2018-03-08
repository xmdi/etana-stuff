all:
	@g++ DECAT.cpp -o DECAT -Ofast -std=c++14
test:
	@./DECAT
