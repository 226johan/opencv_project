#include"label_read.h"

void LabelObj::FileParse()
{
	std::ifstream fp(file_name_);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classes_names_.push_back(name);
	}
	fp.close();
}