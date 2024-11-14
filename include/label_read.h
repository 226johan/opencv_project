#pragma once
#ifndef LABEL_READ_H
#define LABEL_READ_H
#include"config.h"
#include<fstream>


class LabelObj {
public:
	LabelObj() = default;
	explicit LabelObj(const string file_name) :file_name_(file_name) { FileParse(); }
	void FileRead(const string file_name) { file_name_ = file_name; FileParse();}
	void FileParse();
	vector<std::string> ClassseGet() {  return classes_names_; }

	~LabelObj() = default;
	vector<std::string> classes_names_;
	string file_name_;


};

#endif // !LABEL_READ_H

