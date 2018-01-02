#ifndef COMMENT
#define COMMENT
#include<vector>
#include<iostream>
#include<bits\stdc++.h>
using namespace std;

extern bool InitComma();
//extern vector<string> dictionary;
extern vector<string> keybase;
extern vector<unsigned int> keycnt;
extern unsigned int mstFreq;

// ��ȡν�״���Լ����������ϵ��ṹ�еı�����Ϣ��
class Comment
{
public:
	Comment(const string &, bool expandKeyBase = true);
	Comment();
	~Comment();
	void showComment();
	// ��ϵ��ṹ��
	void deConstruct(bool expand = true);
	unsigned int getCommentSize();
	vector<string> getCommmentWord();

private:
	vector<string> comment;
	string level;

	bool isBeAct(const string &str);
};

// several sentences in the one user's comment;
class SerialComment {
public:
	SerialComment();
	SerialComment(const string &, const bool &expandKeyBase = true);
	~SerialComment();
	vector<double> getDigits();
	double getLevel();

private:
	string level;
	void showMsg();

	vector<Comment> usermsg;
};

#endif // COMMENT
