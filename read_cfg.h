 
/*
这里 #define COMMENT '#'是宏定义的方式，易于扩展和重构，增加程序的效率;
下面的两句定义了两个函数，一个是读取配置信息的函数，
	另外一个是打印所有配置信息的函数，实参中的 & 是引用，这样就不是简单的赋值，而是修改实参同时也在修改传递进来的值。
使用COMMENT_CHAR更为清晰易懂一些。
这里定义了一些主要的函数，是为了在main.cpp使用，而read_cfg.cpp中的其他函数都是实现，而非接口。
*/
 
 
#ifndef __GET_CONFIG_H__
#define __GET_CONFIG_H__
 
 
#include <string>
#include <map>
#include <iostream>
using namespace std;
 
#define COMMENT_CHAR '#'
 
bool ReadConfig(const string & filename, map <string, string> & m);
void PrintConfig(const map<string, string> & m);
string FindInConfig(map<string, string> m, string key, string defaultVal);

#endif //
 

