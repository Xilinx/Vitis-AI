#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <unistd.h>

/* global data */
int    apply_filter = 0;
double s0min = 20;
double s0max = 200;
double v0min = 0.1;
double v0max = 2;
std::string tc_file;
std::string ql_file;
std::string op_file;


int get_value(std::string token, double* value, int line_num)
{
    std::istringstream iss_token(token);
    std::string parm;
    if (!std::getline(iss_token, parm, '='))
    {
        std::cout << "ERROR(" << line_num << "): incorrectly formed token: " << token << std::endl;
        return false;
    }

    std::string x;
    if (!std::getline(iss_token, x, '='))
    {
        std::cout << "ERROR(" << line_num << "): incorrectly formed token: " << token << std::endl;
        return false;
    }

    try
    {
        *value = std::stod(x);
    }
    catch (const std::exception& e)
    {
        std::cout << "ERROR (" << line_num << "): not a number: " << value << " : exception: " << e.what() << std::endl;
        return false;
    }

    return true;
}

int token_is(std::string token, std::string match, int line_num)
{
    std::istringstream iss_token(token);
    std::string parm;
    if (!std::getline(iss_token, parm, '='))
    {
        std::cout << "ERROR(" << line_num << "): incorrectly formed token: " << token << std::endl;
        return false;
    }

    if (parm.compare(match) == 0)
    {
        return 1;
    }
    return 0;
}


int filter_line(std::string line, int n)
{
    std::istringstream iss_line(line);
    std::string token;
    if (!apply_filter)
    {
        return 0;
    }

    while (std::getline(iss_line, token, ' '))
    {
        if (token_is(token, "s0", n))
        {
            double value;
            if (get_value(token, &value, n))
            {
                if (value < s0min || value > s0max)
                {
                    return 1;
                }
            }
        }
        if (token_is(token, "v0", n))
        {
            double value;
            if (get_value(token, &value, n))
            {
                if (value < v0min || value > v0max)
                {
                    return 1;
                }
            }
        }
    }
    return 0;
}

int process_file(std::string tc_file, std::string ql_file, std::string op_file)
{
    std::ifstream tc_ifs(tc_file, std::ifstream::in);
    if (!tc_ifs.is_open())
    {
        std::cout << "ERROR: Failed to open file:" << tc_file << std::endl;
        return false;
    }

    std::ifstream ql_ifs(ql_file, std::ifstream::in);
    if (!ql_ifs.is_open())
    {
        std::cout << "ERROR: Failed to open file:" << ql_file << std::endl;
        return false;
    }

    std::ofstream ofs(op_file, std::ios::out);
    if (!ofs.is_open())
    {
        std::cout << "ERROR: Failed to open file:" << op_file << std::endl;
        return false;
    }

    std::string tc_line;
    std::string ql_line;
    int n = 0;
    while (std::getline(ql_ifs, ql_line))
    {
        // generated in windows so remove \r
        if (!ql_line.empty() && ql_line[ql_line.size() - 1] == '\r')
        {
            ql_line.erase(ql_line.size() - 1);
        }

        std::istringstream iss_ql_line(ql_line);
        std::string ql_token;
        while (std::getline(iss_ql_line, ql_token, ','))
        {
            if (std::getline(tc_ifs, tc_line))
            {
                n++;
                if (!filter_line(tc_line, n))
                {
                    ofs << tc_line;
                    ofs << " exp=" << ql_token << std::endl;
                }
            }
        }
    }
}

void usage(char* name)
{
    std::cout << std::endl;
    std::cout << "Usage: " << name << " [-s<s0min> -S<s0max> -v<v0min> -V<v0max>] <tc_file> <ql_file> <op_file>" << std::endl;
    std::cout << "       s0/v0/max/min are filters for test cases (default none; all values used)" << std::endl;
    std::cout << std::endl;
}

int process_args(int argc, char** argv)
{
    int ret = 1; /* OK */

	int opt = 0;
	while ((opt = getopt(argc, argv, "s:S:v:V:")) != -1)
	{
		switch (opt)
		{
			case 's':
				s0min = atof(optarg);
                apply_filter = 1;
				break;
			case 'S':
				s0max = atof(optarg);
                apply_filter = 1;
				break;
			case 'v':
				v0min = atof(optarg);
                apply_filter = 1;
				break;
			case 'V':
				v0max = atof(optarg);
                apply_filter = 1;
				break;
            case 'h':
            default:
                usage(argv[0]);
            	ret = 0;
				break;
        }
    }

    if (ret)
    {
        if (optind + 3 != argc)
        {
            usage(argv[0]);
            ret = 0;
        }
        else
        {
            tc_file = argv[optind++];
            ql_file = argv[optind++];
            op_file = argv[optind++];
        }
    }

    return ret;
}

int main(int argc, char** argv)
{
    if (!process_args(argc, argv))
    {
        std::cout << "ERROR: failed to process command line args" << std::endl;
        return 1;
    }

    if(!process_file(tc_file, ql_file, op_file))
    {
        std::cout << "ERROR: failed to process file" << std::endl;
    }

    return 0;
}


