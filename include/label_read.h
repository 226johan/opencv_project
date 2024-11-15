#pragma once
#ifndef LABEL_READ_H
#define LABEL_READ_H
/**
*	@file label_read.h
*	@brief 文件读取解析
*	Datails.
*
*	@author johan
*	@email 2806762759@qq.com
*	@data 2024.11.14
*/

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

