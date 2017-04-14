namespace common {
	namespace compression {

struct sFastFibonacci2MapRecord
{
	int oldState;
	unsigned char readByte;
	int newState;
	unsigned char writeCount;
	 unsigned int code0;
	 unsigned int bits0;
	 unsigned int out0;
	 unsigned int code1;
	 unsigned int bits1;
	 unsigned int out1;
	 unsigned int code2;
	 unsigned int bits2;
	 unsigned int out2;
	 unsigned int code3;
	 unsigned int bits3;
	 unsigned int out3;
	 unsigned int code4;
	 unsigned int bits4;
	 unsigned int out4;
};
const struct sFastFibonacci2MapRecord Fibonacci2MapTable[] = 
{
0,0,0,1 ,0,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,1,0,1 ,1,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,2,0,1 ,2,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,3,0,2 ,1,1,1 ,0,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,4,0,1 ,3,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,5,0,1 ,4,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,6,0,2 ,2,2,1 ,0,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,7,0,2 ,1,1,1 ,1,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,8,0,1 ,5,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,9,0,1 ,6,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,10,0,1 ,7,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,11,0,2 ,1,1,1 ,2,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,12,0,2 ,3,3,1 ,0,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,13,0,2 ,4,3,1 ,0,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,14,0,2 ,2,2,1 ,1,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,15,0,3 ,1,1,1 ,1,1,1 ,0,4,0 ,0,0,0 ,0,0,0
,0,16,0,1 ,8,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,17,0,1 ,9,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,18,0,1 ,10,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,19,0,2 ,1,1,1 ,3,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,20,0,1 ,11,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,21,0,1 ,12,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,22,0,2 ,2,2,1 ,2,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,23,0,2 ,1,1,1 ,4,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,24,0,2 ,5,4,1 ,0,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,25,0,2 ,6,4,1 ,0,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,26,0,2 ,7,4,1 ,0,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,27,0,3 ,1,1,1 ,2,2,1 ,0,3,0 ,0,0,0 ,0,0,0
,0,28,0,2 ,3,3,1 ,1,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,29,0,2 ,4,3,1 ,1,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,30,0,3 ,2,2,1 ,1,1,1 ,0,3,0 ,0,0,0 ,0,0,0
,0,31,0,3 ,1,1,1 ,1,1,1 ,1,4,0 ,0,0,0 ,0,0,0
,0,32,0,1 ,13,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,33,0,1 ,14,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,34,0,1 ,15,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,35,0,2 ,1,1,1 ,5,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,36,0,1 ,16,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,37,0,1 ,17,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,38,0,2 ,2,2,1 ,3,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,39,0,2 ,1,1,1 ,6,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,40,0,1 ,18,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,41,0,1 ,19,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,42,0,1 ,20,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,43,0,2 ,1,1,1 ,7,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,44,0,2 ,3,3,1 ,2,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,45,0,2 ,4,3,1 ,2,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,46,0,2 ,2,2,1 ,4,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,47,0,3 ,1,1,1 ,1,1,1 ,2,4,0 ,0,0,0 ,0,0,0
,0,48,0,2 ,8,5,1 ,0,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,49,0,2 ,9,5,1 ,0,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,50,0,2 ,10,5,1 ,0,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,51,0,3 ,1,1,1 ,3,3,1 ,0,2,0 ,0,0,0 ,0,0,0
,0,52,0,2 ,11,5,1 ,0,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,53,0,2 ,12,5,1 ,0,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,54,0,3 ,2,2,1 ,2,2,1 ,0,2,0 ,0,0,0 ,0,0,0
,0,55,0,3 ,1,1,1 ,4,3,1 ,0,2,0 ,0,0,0 ,0,0,0
,0,56,0,2 ,5,4,1 ,1,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,57,0,2 ,6,4,1 ,1,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,58,0,2 ,7,4,1 ,1,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,59,0,3 ,1,1,1 ,2,2,1 ,1,3,0 ,0,0,0 ,0,0,0
,0,60,0,3 ,3,3,1 ,1,1,1 ,0,2,0 ,0,0,0 ,0,0,0
,0,61,0,3 ,4,3,1 ,1,1,1 ,0,2,0 ,0,0,0 ,0,0,0
,0,62,0,3 ,2,2,1 ,1,1,1 ,1,3,0 ,0,0,0 ,0,0,0
,0,63,0,4 ,1,1,1 ,1,1,1 ,1,1,1 ,0,2,0 ,0,0,0
,0,64,0,1 ,21,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,65,0,1 ,22,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,66,0,1 ,23,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,67,0,2 ,1,1,1 ,8,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,68,0,1 ,24,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,69,0,1 ,25,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,70,0,2 ,2,2,1 ,5,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,71,0,2 ,1,1,1 ,9,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,72,0,1 ,26,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,73,0,1 ,27,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,74,0,1 ,28,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,75,0,2 ,1,1,1 ,10,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,76,0,2 ,3,3,1 ,3,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,77,0,2 ,4,3,1 ,3,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,78,0,2 ,2,2,1 ,6,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,79,0,3 ,1,1,1 ,1,1,1 ,3,4,0 ,0,0,0 ,0,0,0
,0,80,0,1 ,29,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,81,0,1 ,30,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,82,0,1 ,31,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,83,0,2 ,1,1,1 ,11,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,84,0,1 ,32,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,85,0,1 ,33,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,86,0,2 ,2,2,1 ,7,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,87,0,2 ,1,1,1 ,12,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,88,0,2 ,5,4,1 ,2,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,89,0,2 ,6,4,1 ,2,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,90,0,2 ,7,4,1 ,2,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,91,0,3 ,1,1,1 ,2,2,1 ,2,3,0 ,0,0,0 ,0,0,0
,0,92,0,2 ,3,3,1 ,4,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,93,0,2 ,4,3,1 ,4,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,94,0,3 ,2,2,1 ,1,1,1 ,2,3,0 ,0,0,0 ,0,0,0
,0,95,0,3 ,1,1,1 ,1,1,1 ,4,4,0 ,0,0,0 ,0,0,0
,0,96,0,2 ,13,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,97,0,2 ,14,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,98,0,2 ,15,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,99,0,3 ,1,1,1 ,5,4,1 ,0,1,0 ,0,0,0 ,0,0,0
,0,100,0,2 ,16,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,101,0,2 ,17,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,102,0,3 ,2,2,1 ,3,3,1 ,0,1,0 ,0,0,0 ,0,0,0
,0,103,0,3 ,1,1,1 ,6,4,1 ,0,1,0 ,0,0,0 ,0,0,0
,0,104,0,2 ,18,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,105,0,2 ,19,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,106,0,2 ,20,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,107,0,3 ,1,1,1 ,7,4,1 ,0,1,0 ,0,0,0 ,0,0,0
,0,108,0,3 ,3,3,1 ,2,2,1 ,0,1,0 ,0,0,0 ,0,0,0
,0,109,0,3 ,4,3,1 ,2,2,1 ,0,1,0 ,0,0,0 ,0,0,0
,0,110,0,3 ,2,2,1 ,4,3,1 ,0,1,0 ,0,0,0 ,0,0,0
,0,111,0,4 ,1,1,1 ,1,1,1 ,2,2,1 ,0,1,0 ,0,0,0
,0,112,0,2 ,8,5,1 ,1,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,113,0,2 ,9,5,1 ,1,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,114,0,2 ,10,5,1 ,1,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,115,0,3 ,1,1,1 ,3,3,1 ,1,2,0 ,0,0,0 ,0,0,0
,0,116,0,2 ,11,5,1 ,1,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,117,0,2 ,12,5,1 ,1,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,118,0,3 ,2,2,1 ,2,2,1 ,1,2,0 ,0,0,0 ,0,0,0
,0,119,0,3 ,1,1,1 ,4,3,1 ,1,2,0 ,0,0,0 ,0,0,0
,0,120,0,3 ,5,4,1 ,1,1,1 ,0,1,0 ,0,0,0 ,0,0,0
,0,121,0,3 ,6,4,1 ,1,1,1 ,0,1,0 ,0,0,0 ,0,0,0
,0,122,0,3 ,7,4,1 ,1,1,1 ,0,1,0 ,0,0,0 ,0,0,0
,0,123,0,4 ,1,1,1 ,2,2,1 ,1,1,1 ,0,1,0 ,0,0,0
,0,124,0,3 ,3,3,1 ,1,1,1 ,1,2,0 ,0,0,0 ,0,0,0
,0,125,0,3 ,4,3,1 ,1,1,1 ,1,2,0 ,0,0,0 ,0,0,0
,0,126,0,4 ,2,2,1 ,1,1,1 ,1,1,1 ,0,1,0 ,0,0,0
,0,127,0,4 ,1,1,1 ,1,1,1 ,1,1,1 ,1,2,0 ,0,0,0
,0,128,256,1 ,34,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,129,256,1 ,35,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,130,256,1 ,36,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,131,256,2 ,1,1,1 ,13,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,132,256,1 ,37,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,133,256,1 ,38,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,134,256,2 ,2,2,1 ,8,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,135,256,2 ,1,1,1 ,14,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,136,256,1 ,39,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,137,256,1 ,40,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,138,256,1 ,41,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,139,256,2 ,1,1,1 ,15,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,140,256,2 ,3,3,1 ,5,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,141,256,2 ,4,3,1 ,5,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,142,256,2 ,2,2,1 ,9,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,143,256,3 ,1,1,1 ,1,1,1 ,5,4,0 ,0,0,0 ,0,0,0
,0,144,256,1 ,42,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,145,256,1 ,43,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,146,256,1 ,44,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,147,256,2 ,1,1,1 ,16,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,148,256,1 ,45,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,149,256,1 ,46,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,150,256,2 ,2,2,1 ,10,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,151,256,2 ,1,1,1 ,17,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,152,256,2 ,5,4,1 ,3,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,153,256,2 ,6,4,1 ,3,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,154,256,2 ,7,4,1 ,3,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,155,256,3 ,1,1,1 ,2,2,1 ,3,3,0 ,0,0,0 ,0,0,0
,0,156,256,2 ,3,3,1 ,6,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,157,256,2 ,4,3,1 ,6,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,158,256,3 ,2,2,1 ,1,1,1 ,3,3,0 ,0,0,0 ,0,0,0
,0,159,256,3 ,1,1,1 ,1,1,1 ,6,4,0 ,0,0,0 ,0,0,0
,0,160,256,1 ,47,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,161,256,1 ,48,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,162,256,1 ,49,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,163,256,2 ,1,1,1 ,18,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,164,256,1 ,50,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,165,256,1 ,51,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,166,256,2 ,2,2,1 ,11,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,167,256,2 ,1,1,1 ,19,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,168,256,1 ,52,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,169,256,1 ,53,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,170,256,1 ,54,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,171,256,2 ,1,1,1 ,20,6,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,172,256,2 ,3,3,1 ,7,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,173,256,2 ,4,3,1 ,7,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,174,256,2 ,2,2,1 ,12,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,175,256,3 ,1,1,1 ,1,1,1 ,7,4,0 ,0,0,0 ,0,0,0
,0,176,256,2 ,8,5,1 ,2,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,177,256,2 ,9,5,1 ,2,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,178,256,2 ,10,5,1 ,2,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,179,256,3 ,1,1,1 ,3,3,1 ,2,2,0 ,0,0,0 ,0,0,0
,0,180,256,2 ,11,5,1 ,2,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,181,256,2 ,12,5,1 ,2,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,182,256,3 ,2,2,1 ,2,2,1 ,2,2,0 ,0,0,0 ,0,0,0
,0,183,256,3 ,1,1,1 ,4,3,1 ,2,2,0 ,0,0,0 ,0,0,0
,0,184,256,2 ,5,4,1 ,4,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,185,256,2 ,6,4,1 ,4,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,186,256,2 ,7,4,1 ,4,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,187,256,3 ,1,1,1 ,2,2,1 ,4,3,0 ,0,0,0 ,0,0,0
,0,188,256,3 ,3,3,1 ,1,1,1 ,2,2,0 ,0,0,0 ,0,0,0
,0,189,256,3 ,4,3,1 ,1,1,1 ,2,2,0 ,0,0,0 ,0,0,0
,0,190,256,3 ,2,2,1 ,1,1,1 ,4,3,0 ,0,0,0 ,0,0,0
,0,191,256,4 ,1,1,1 ,1,1,1 ,1,1,1 ,2,2,0 ,0,0,0
,0,192,0,1 ,21,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,193,0,1 ,22,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,194,0,1 ,23,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,195,0,2 ,1,1,1 ,8,5,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,196,0,1 ,24,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,197,0,1 ,25,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,198,0,2 ,2,2,1 ,5,4,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,199,0,2 ,1,1,1 ,9,5,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,200,0,1 ,26,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,201,0,1 ,27,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,202,0,1 ,28,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,203,0,2 ,1,1,1 ,10,5,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,204,0,2 ,3,3,1 ,3,3,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,205,0,2 ,4,3,1 ,3,3,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,206,0,2 ,2,2,1 ,6,4,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,207,0,3 ,1,1,1 ,1,1,1 ,3,3,1 ,0,0,0 ,0,0,0
,0,208,0,1 ,29,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,209,0,1 ,30,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,210,0,1 ,31,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,211,0,2 ,1,1,1 ,11,5,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,212,0,1 ,32,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,213,0,1 ,33,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,214,0,2 ,2,2,1 ,7,4,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,215,0,2 ,1,1,1 ,12,5,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,216,0,2 ,5,4,1 ,2,2,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,217,0,2 ,6,4,1 ,2,2,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,218,0,2 ,7,4,1 ,2,2,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,219,0,3 ,1,1,1 ,2,2,1 ,2,2,1 ,0,0,0 ,0,0,0
,0,220,0,2 ,3,3,1 ,4,3,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,221,0,2 ,4,3,1 ,4,3,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,222,0,3 ,2,2,1 ,1,1,1 ,2,2,1 ,0,0,0 ,0,0,0
,0,223,0,3 ,1,1,1 ,1,1,1 ,4,3,1 ,0,0,0 ,0,0,0
,0,224,256,2 ,13,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,225,256,2 ,14,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,226,256,2 ,15,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,227,256,3 ,1,1,1 ,5,4,1 ,1,1,0 ,0,0,0 ,0,0,0
,0,228,256,2 ,16,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,229,256,2 ,17,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,230,256,3 ,2,2,1 ,3,3,1 ,1,1,0 ,0,0,0 ,0,0,0
,0,231,256,3 ,1,1,1 ,6,4,1 ,1,1,0 ,0,0,0 ,0,0,0
,0,232,256,2 ,18,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,233,256,2 ,19,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,234,256,2 ,20,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,0,235,256,3 ,1,1,1 ,7,4,1 ,1,1,0 ,0,0,0 ,0,0,0
,0,236,256,3 ,3,3,1 ,2,2,1 ,1,1,0 ,0,0,0 ,0,0,0
,0,237,256,3 ,4,3,1 ,2,2,1 ,1,1,0 ,0,0,0 ,0,0,0
,0,238,256,3 ,2,2,1 ,4,3,1 ,1,1,0 ,0,0,0 ,0,0,0
,0,239,256,4 ,1,1,1 ,1,1,1 ,2,2,1 ,1,1,0 ,0,0,0
,0,240,0,2 ,8,5,1 ,1,1,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,241,0,2 ,9,5,1 ,1,1,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,242,0,2 ,10,5,1 ,1,1,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,243,0,3 ,1,1,1 ,3,3,1 ,1,1,1 ,0,0,0 ,0,0,0
,0,244,0,2 ,11,5,1 ,1,1,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,245,0,2 ,12,5,1 ,1,1,1 ,0,0,0 ,0,0,0 ,0,0,0
,0,246,0,3 ,2,2,1 ,2,2,1 ,1,1,1 ,0,0,0 ,0,0,0
,0,247,0,3 ,1,1,1 ,4,3,1 ,1,1,1 ,0,0,0 ,0,0,0
,0,248,256,3 ,5,4,1 ,1,1,1 ,1,1,0 ,0,0,0 ,0,0,0
,0,249,256,3 ,6,4,1 ,1,1,1 ,1,1,0 ,0,0,0 ,0,0,0
,0,250,256,3 ,7,4,1 ,1,1,1 ,1,1,0 ,0,0,0 ,0,0,0
,0,251,256,4 ,1,1,1 ,2,2,1 ,1,1,1 ,1,1,0 ,0,0,0
,0,252,0,3 ,3,3,1 ,1,1,1 ,1,1,1 ,0,0,0 ,0,0,0
,0,253,0,3 ,4,3,1 ,1,1,1 ,1,1,1 ,0,0,0 ,0,0,0
,0,254,256,4 ,2,2,1 ,1,1,1 ,1,1,1 ,1,1,0 ,0,0,0
,0,255,0,4 ,1,1,1 ,1,1,1 ,1,1,1 ,1,1,1 ,0,0,0
,256,0,0,1 ,0,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,1,0,2 ,0,0,1 ,0,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,2,0,1 ,2,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,3,0,2 ,0,0,1 ,1,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,4,0,1 ,3,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,5,0,2 ,0,0,1 ,2,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,6,0,2 ,2,2,1 ,0,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,7,0,3 ,0,0,1 ,1,1,1 ,0,5,0 ,0,0,0 ,0,0,0
,256,8,0,1 ,5,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,9,0,2 ,0,0,1 ,3,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,10,0,1 ,7,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,11,0,2 ,0,0,1 ,4,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,12,0,2 ,3,3,1 ,0,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,13,0,3 ,0,0,1 ,2,2,1 ,0,4,0 ,0,0,0 ,0,0,0
,256,14,0,2 ,2,2,1 ,1,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,15,0,3 ,0,0,1 ,1,1,1 ,1,5,0 ,0,0,0 ,0,0,0
,256,16,0,1 ,8,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,17,0,2 ,0,0,1 ,5,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,18,0,1 ,10,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,19,0,2 ,0,0,1 ,6,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,20,0,1 ,11,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,21,0,2 ,0,0,1 ,7,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,22,0,2 ,2,2,1 ,2,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,23,0,3 ,0,0,1 ,1,1,1 ,2,5,0 ,0,0,0 ,0,0,0
,256,24,0,2 ,5,4,1 ,0,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,25,0,3 ,0,0,1 ,3,3,1 ,0,3,0 ,0,0,0 ,0,0,0
,256,26,0,2 ,7,4,1 ,0,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,27,0,3 ,0,0,1 ,4,3,1 ,0,3,0 ,0,0,0 ,0,0,0
,256,28,0,2 ,3,3,1 ,1,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,29,0,3 ,0,0,1 ,2,2,1 ,1,4,0 ,0,0,0 ,0,0,0
,256,30,0,3 ,2,2,1 ,1,1,1 ,0,3,0 ,0,0,0 ,0,0,0
,256,31,0,4 ,0,0,1 ,1,1,1 ,1,1,1 ,0,3,0 ,0,0,0
,256,32,0,1 ,13,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,33,0,2 ,0,0,1 ,8,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,34,0,1 ,15,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,35,0,2 ,0,0,1 ,9,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,36,0,1 ,16,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,37,0,2 ,0,0,1 ,10,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,38,0,2 ,2,2,1 ,3,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,39,0,3 ,0,0,1 ,1,1,1 ,3,5,0 ,0,0,0 ,0,0,0
,256,40,0,1 ,18,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,41,0,2 ,0,0,1 ,11,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,42,0,1 ,20,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,43,0,2 ,0,0,1 ,12,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,44,0,2 ,3,3,1 ,2,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,45,0,3 ,0,0,1 ,2,2,1 ,2,4,0 ,0,0,0 ,0,0,0
,256,46,0,2 ,2,2,1 ,4,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,47,0,3 ,0,0,1 ,1,1,1 ,4,5,0 ,0,0,0 ,0,0,0
,256,48,0,2 ,8,5,1 ,0,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,49,0,3 ,0,0,1 ,5,4,1 ,0,2,0 ,0,0,0 ,0,0,0
,256,50,0,2 ,10,5,1 ,0,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,51,0,3 ,0,0,1 ,6,4,1 ,0,2,0 ,0,0,0 ,0,0,0
,256,52,0,2 ,11,5,1 ,0,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,53,0,3 ,0,0,1 ,7,4,1 ,0,2,0 ,0,0,0 ,0,0,0
,256,54,0,3 ,2,2,1 ,2,2,1 ,0,2,0 ,0,0,0 ,0,0,0
,256,55,0,4 ,0,0,1 ,1,1,1 ,2,2,1 ,0,2,0 ,0,0,0
,256,56,0,2 ,5,4,1 ,1,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,57,0,3 ,0,0,1 ,3,3,1 ,1,3,0 ,0,0,0 ,0,0,0
,256,58,0,2 ,7,4,1 ,1,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,59,0,3 ,0,0,1 ,4,3,1 ,1,3,0 ,0,0,0 ,0,0,0
,256,60,0,3 ,3,3,1 ,1,1,1 ,0,2,0 ,0,0,0 ,0,0,0
,256,61,0,4 ,0,0,1 ,2,2,1 ,1,1,1 ,0,2,0 ,0,0,0
,256,62,0,3 ,2,2,1 ,1,1,1 ,1,3,0 ,0,0,0 ,0,0,0
,256,63,0,4 ,0,0,1 ,1,1,1 ,1,1,1 ,1,3,0 ,0,0,0
,256,64,0,1 ,21,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,65,0,2 ,0,0,1 ,13,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,66,0,1 ,23,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,67,0,2 ,0,0,1 ,14,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,68,0,1 ,24,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,69,0,2 ,0,0,1 ,15,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,70,0,2 ,2,2,1 ,5,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,71,0,3 ,0,0,1 ,1,1,1 ,5,5,0 ,0,0,0 ,0,0,0
,256,72,0,1 ,26,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,73,0,2 ,0,0,1 ,16,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,74,0,1 ,28,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,75,0,2 ,0,0,1 ,17,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,76,0,2 ,3,3,1 ,3,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,77,0,3 ,0,0,1 ,2,2,1 ,3,4,0 ,0,0,0 ,0,0,0
,256,78,0,2 ,2,2,1 ,6,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,79,0,3 ,0,0,1 ,1,1,1 ,6,5,0 ,0,0,0 ,0,0,0
,256,80,0,1 ,29,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,81,0,2 ,0,0,1 ,18,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,82,0,1 ,31,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,83,0,2 ,0,0,1 ,19,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,84,0,1 ,32,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,85,0,2 ,0,0,1 ,20,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,86,0,2 ,2,2,1 ,7,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,87,0,3 ,0,0,1 ,1,1,1 ,7,5,0 ,0,0,0 ,0,0,0
,256,88,0,2 ,5,4,1 ,2,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,89,0,3 ,0,0,1 ,3,3,1 ,2,3,0 ,0,0,0 ,0,0,0
,256,90,0,2 ,7,4,1 ,2,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,91,0,3 ,0,0,1 ,4,3,1 ,2,3,0 ,0,0,0 ,0,0,0
,256,92,0,2 ,3,3,1 ,4,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,93,0,3 ,0,0,1 ,2,2,1 ,4,4,0 ,0,0,0 ,0,0,0
,256,94,0,3 ,2,2,1 ,1,1,1 ,2,3,0 ,0,0,0 ,0,0,0
,256,95,0,4 ,0,0,1 ,1,1,1 ,1,1,1 ,2,3,0 ,0,0,0
,256,96,0,2 ,13,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,97,0,3 ,0,0,1 ,8,5,1 ,0,1,0 ,0,0,0 ,0,0,0
,256,98,0,2 ,15,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,99,0,3 ,0,0,1 ,9,5,1 ,0,1,0 ,0,0,0 ,0,0,0
,256,100,0,2 ,16,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,101,0,3 ,0,0,1 ,10,5,1 ,0,1,0 ,0,0,0 ,0,0,0
,256,102,0,3 ,2,2,1 ,3,3,1 ,0,1,0 ,0,0,0 ,0,0,0
,256,103,0,4 ,0,0,1 ,1,1,1 ,3,3,1 ,0,1,0 ,0,0,0
,256,104,0,2 ,18,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,105,0,3 ,0,0,1 ,11,5,1 ,0,1,0 ,0,0,0 ,0,0,0
,256,106,0,2 ,20,6,1 ,0,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,107,0,3 ,0,0,1 ,12,5,1 ,0,1,0 ,0,0,0 ,0,0,0
,256,108,0,3 ,3,3,1 ,2,2,1 ,0,1,0 ,0,0,0 ,0,0,0
,256,109,0,4 ,0,0,1 ,2,2,1 ,2,2,1 ,0,1,0 ,0,0,0
,256,110,0,3 ,2,2,1 ,4,3,1 ,0,1,0 ,0,0,0 ,0,0,0
,256,111,0,4 ,0,0,1 ,1,1,1 ,4,3,1 ,0,1,0 ,0,0,0
,256,112,0,2 ,8,5,1 ,1,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,113,0,3 ,0,0,1 ,5,4,1 ,1,2,0 ,0,0,0 ,0,0,0
,256,114,0,2 ,10,5,1 ,1,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,115,0,3 ,0,0,1 ,6,4,1 ,1,2,0 ,0,0,0 ,0,0,0
,256,116,0,2 ,11,5,1 ,1,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,117,0,3 ,0,0,1 ,7,4,1 ,1,2,0 ,0,0,0 ,0,0,0
,256,118,0,3 ,2,2,1 ,2,2,1 ,1,2,0 ,0,0,0 ,0,0,0
,256,119,0,4 ,0,0,1 ,1,1,1 ,2,2,1 ,1,2,0 ,0,0,0
,256,120,0,3 ,5,4,1 ,1,1,1 ,0,1,0 ,0,0,0 ,0,0,0
,256,121,0,4 ,0,0,1 ,3,3,1 ,1,1,1 ,0,1,0 ,0,0,0
,256,122,0,3 ,7,4,1 ,1,1,1 ,0,1,0 ,0,0,0 ,0,0,0
,256,123,0,4 ,0,0,1 ,4,3,1 ,1,1,1 ,0,1,0 ,0,0,0
,256,124,0,3 ,3,3,1 ,1,1,1 ,1,2,0 ,0,0,0 ,0,0,0
,256,125,0,4 ,0,0,1 ,2,2,1 ,1,1,1 ,1,2,0 ,0,0,0
,256,126,0,4 ,2,2,1 ,1,1,1 ,1,1,1 ,0,1,0 ,0,0,0
,256,127,0,5 ,0,0,1 ,1,1,1 ,1,1,1 ,1,1,1 ,0,1,0
,256,128,256,1 ,34,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,129,256,2 ,0,0,1 ,21,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,130,256,1 ,36,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,131,256,2 ,0,0,1 ,22,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,132,256,1 ,37,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,133,256,2 ,0,0,1 ,23,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,134,256,2 ,2,2,1 ,8,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,135,256,3 ,0,0,1 ,1,1,1 ,8,5,0 ,0,0,0 ,0,0,0
,256,136,256,1 ,39,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,137,256,2 ,0,0,1 ,24,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,138,256,1 ,41,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,139,256,2 ,0,0,1 ,25,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,140,256,2 ,3,3,1 ,5,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,141,256,3 ,0,0,1 ,2,2,1 ,5,4,0 ,0,0,0 ,0,0,0
,256,142,256,2 ,2,2,1 ,9,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,143,256,3 ,0,0,1 ,1,1,1 ,9,5,0 ,0,0,0 ,0,0,0
,256,144,256,1 ,42,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,145,256,2 ,0,0,1 ,26,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,146,256,1 ,44,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,147,256,2 ,0,0,1 ,27,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,148,256,1 ,45,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,149,256,2 ,0,0,1 ,28,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,150,256,2 ,2,2,1 ,10,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,151,256,3 ,0,0,1 ,1,1,1 ,10,5,0 ,0,0,0 ,0,0,0
,256,152,256,2 ,5,4,1 ,3,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,153,256,3 ,0,0,1 ,3,3,1 ,3,3,0 ,0,0,0 ,0,0,0
,256,154,256,2 ,7,4,1 ,3,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,155,256,3 ,0,0,1 ,4,3,1 ,3,3,0 ,0,0,0 ,0,0,0
,256,156,256,2 ,3,3,1 ,6,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,157,256,3 ,0,0,1 ,2,2,1 ,6,4,0 ,0,0,0 ,0,0,0
,256,158,256,3 ,2,2,1 ,1,1,1 ,3,3,0 ,0,0,0 ,0,0,0
,256,159,256,4 ,0,0,1 ,1,1,1 ,1,1,1 ,3,3,0 ,0,0,0
,256,160,256,1 ,47,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,161,256,2 ,0,0,1 ,29,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,162,256,1 ,49,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,163,256,2 ,0,0,1 ,30,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,164,256,1 ,50,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,165,256,2 ,0,0,1 ,31,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,166,256,2 ,2,2,1 ,11,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,167,256,3 ,0,0,1 ,1,1,1 ,11,5,0 ,0,0,0 ,0,0,0
,256,168,256,1 ,52,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,169,256,2 ,0,0,1 ,32,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,170,256,1 ,54,8,0 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,171,256,2 ,0,0,1 ,33,7,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,172,256,2 ,3,3,1 ,7,4,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,173,256,3 ,0,0,1 ,2,2,1 ,7,4,0 ,0,0,0 ,0,0,0
,256,174,256,2 ,2,2,1 ,12,5,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,175,256,3 ,0,0,1 ,1,1,1 ,12,5,0 ,0,0,0 ,0,0,0
,256,176,256,2 ,8,5,1 ,2,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,177,256,3 ,0,0,1 ,5,4,1 ,2,2,0 ,0,0,0 ,0,0,0
,256,178,256,2 ,10,5,1 ,2,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,179,256,3 ,0,0,1 ,6,4,1 ,2,2,0 ,0,0,0 ,0,0,0
,256,180,256,2 ,11,5,1 ,2,2,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,181,256,3 ,0,0,1 ,7,4,1 ,2,2,0 ,0,0,0 ,0,0,0
,256,182,256,3 ,2,2,1 ,2,2,1 ,2,2,0 ,0,0,0 ,0,0,0
,256,183,256,4 ,0,0,1 ,1,1,1 ,2,2,1 ,2,2,0 ,0,0,0
,256,184,256,2 ,5,4,1 ,4,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,185,256,3 ,0,0,1 ,3,3,1 ,4,3,0 ,0,0,0 ,0,0,0
,256,186,256,2 ,7,4,1 ,4,3,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,187,256,3 ,0,0,1 ,4,3,1 ,4,3,0 ,0,0,0 ,0,0,0
,256,188,256,3 ,3,3,1 ,1,1,1 ,2,2,0 ,0,0,0 ,0,0,0
,256,189,256,4 ,0,0,1 ,2,2,1 ,1,1,1 ,2,2,0 ,0,0,0
,256,190,256,3 ,2,2,1 ,1,1,1 ,4,3,0 ,0,0,0 ,0,0,0
,256,191,256,4 ,0,0,1 ,1,1,1 ,1,1,1 ,4,3,0 ,0,0,0
,256,192,0,1 ,21,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,193,0,2 ,0,0,1 ,13,6,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,194,0,1 ,23,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,195,0,2 ,0,0,1 ,14,6,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,196,0,1 ,24,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,197,0,2 ,0,0,1 ,15,6,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,198,0,2 ,2,2,1 ,5,4,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,199,0,3 ,0,0,1 ,1,1,1 ,5,4,1 ,0,0,0 ,0,0,0
,256,200,0,1 ,26,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,201,0,2 ,0,0,1 ,16,6,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,202,0,1 ,28,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,203,0,2 ,0,0,1 ,17,6,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,204,0,2 ,3,3,1 ,3,3,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,205,0,3 ,0,0,1 ,2,2,1 ,3,3,1 ,0,0,0 ,0,0,0
,256,206,0,2 ,2,2,1 ,6,4,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,207,0,3 ,0,0,1 ,1,1,1 ,6,4,1 ,0,0,0 ,0,0,0
,256,208,0,1 ,29,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,209,0,2 ,0,0,1 ,18,6,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,210,0,1 ,31,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,211,0,2 ,0,0,1 ,19,6,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,212,0,1 ,32,7,1 ,0,0,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,213,0,2 ,0,0,1 ,20,6,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,214,0,2 ,2,2,1 ,7,4,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,215,0,3 ,0,0,1 ,1,1,1 ,7,4,1 ,0,0,0 ,0,0,0
,256,216,0,2 ,5,4,1 ,2,2,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,217,0,3 ,0,0,1 ,3,3,1 ,2,2,1 ,0,0,0 ,0,0,0
,256,218,0,2 ,7,4,1 ,2,2,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,219,0,3 ,0,0,1 ,4,3,1 ,2,2,1 ,0,0,0 ,0,0,0
,256,220,0,2 ,3,3,1 ,4,3,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,221,0,3 ,0,0,1 ,2,2,1 ,4,3,1 ,0,0,0 ,0,0,0
,256,222,0,3 ,2,2,1 ,1,1,1 ,2,2,1 ,0,0,0 ,0,0,0
,256,223,0,4 ,0,0,1 ,1,1,1 ,1,1,1 ,2,2,1 ,0,0,0
,256,224,256,2 ,13,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,225,256,3 ,0,0,1 ,8,5,1 ,1,1,0 ,0,0,0 ,0,0,0
,256,226,256,2 ,15,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,227,256,3 ,0,0,1 ,9,5,1 ,1,1,0 ,0,0,0 ,0,0,0
,256,228,256,2 ,16,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,229,256,3 ,0,0,1 ,10,5,1 ,1,1,0 ,0,0,0 ,0,0,0
,256,230,256,3 ,2,2,1 ,3,3,1 ,1,1,0 ,0,0,0 ,0,0,0
,256,231,256,4 ,0,0,1 ,1,1,1 ,3,3,1 ,1,1,0 ,0,0,0
,256,232,256,2 ,18,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,233,256,3 ,0,0,1 ,11,5,1 ,1,1,0 ,0,0,0 ,0,0,0
,256,234,256,2 ,20,6,1 ,1,1,0 ,0,0,0 ,0,0,0 ,0,0,0
,256,235,256,3 ,0,0,1 ,12,5,1 ,1,1,0 ,0,0,0 ,0,0,0
,256,236,256,3 ,3,3,1 ,2,2,1 ,1,1,0 ,0,0,0 ,0,0,0
,256,237,256,4 ,0,0,1 ,2,2,1 ,2,2,1 ,1,1,0 ,0,0,0
,256,238,256,3 ,2,2,1 ,4,3,1 ,1,1,0 ,0,0,0 ,0,0,0
,256,239,256,4 ,0,0,1 ,1,1,1 ,4,3,1 ,1,1,0 ,0,0,0
,256,240,0,2 ,8,5,1 ,1,1,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,241,0,3 ,0,0,1 ,5,4,1 ,1,1,1 ,0,0,0 ,0,0,0
,256,242,0,2 ,10,5,1 ,1,1,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,243,0,3 ,0,0,1 ,6,4,1 ,1,1,1 ,0,0,0 ,0,0,0
,256,244,0,2 ,11,5,1 ,1,1,1 ,0,0,0 ,0,0,0 ,0,0,0
,256,245,0,3 ,0,0,1 ,7,4,1 ,1,1,1 ,0,0,0 ,0,0,0
,256,246,0,3 ,2,2,1 ,2,2,1 ,1,1,1 ,0,0,0 ,0,0,0
,256,247,0,4 ,0,0,1 ,1,1,1 ,2,2,1 ,1,1,1 ,0,0,0
,256,248,256,3 ,5,4,1 ,1,1,1 ,1,1,0 ,0,0,0 ,0,0,0
,256,249,256,4 ,0,0,1 ,3,3,1 ,1,1,1 ,1,1,0 ,0,0,0
,256,250,256,3 ,7,4,1 ,1,1,1 ,1,1,0 ,0,0,0 ,0,0,0
,256,251,256,4 ,0,0,1 ,4,3,1 ,1,1,1 ,1,1,0 ,0,0,0
,256,252,0,3 ,3,3,1 ,1,1,1 ,1,1,1 ,0,0,0 ,0,0,0
,256,253,0,4 ,0,0,1 ,2,2,1 ,1,1,1 ,1,1,1 ,0,0,0
,256,254,256,4 ,2,2,1 ,1,1,1 ,1,1,1 ,1,1,0 ,0,0,0
,256,255,256,5 ,0,0,1 ,1,1,1 ,1,1,1 ,1,1,1 ,1,1,0
};
}}