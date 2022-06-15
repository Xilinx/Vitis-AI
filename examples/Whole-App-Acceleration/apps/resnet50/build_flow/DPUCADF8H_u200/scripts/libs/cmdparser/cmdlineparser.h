/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
#ifndef CMDLINEPARSER_H_
#define CMDLINEPARSER_H_

#include <map>
#include <string>
#include <vector>

using namespace std;

namespace sda {
namespace utils {

bool is_file(const std::string& name);

/*!
 * Synopsis:
 * 1.Parses the command line passed in from the user and stores all enabled
 *      system options.
 * 2.Prints help for the user if an option is not valid.
 * 3.Stores options and provides a mechanism to read those options
 */
class CmdLineParser {
public:
    class CmdSwitch {
    public:
        CmdSwitch() {}
        CmdSwitch(const CmdSwitch& rhs) {
            copyfrom(rhs);
        }

        void copyfrom(const CmdSwitch& rhs) {
            this->key = rhs.key;
            this->shortcut = rhs.shortcut;
            this->default_value = rhs.default_value;
            this->value = rhs.value;
            this->desc = rhs.desc;
            this->istoggle = rhs.istoggle;
            this->isvalid = rhs.isvalid;
        }

        CmdSwitch& operator=(const CmdSwitch& rhs) {
            this->copyfrom(rhs);
            return *this;
        }
    public:
        string key;
        string shortcut;
        string default_value;
        string value;
        string desc;
        bool istoggle;
        bool isvalid;
    };

public:
    CmdLineParser();
    //CmdLineParser(int argc, char* argv[]);
    virtual ~CmdLineParser();


    bool addSwitch(const CmdSwitch& s);
    bool addSwitch(const string& name, const string& shortcut,
                    const string& desc, const string& default_value = "",
                    bool istoggle = false);

    /*!
     * sets default key to be able to read a 2 argumented call
     */
    bool setDefaultKey(const char* key);

    /*!
     * parse and store command line
     */
    int parse(int argc, char* argv[]);

    /*!
     * retrieve value using a key
     */
    string value(const char* key);

    int value_to_int(const char* key);


    double value_to_double(const char* key);

    /*!
     * Returns true if a valid value is supplied by user
     */
    bool isValid(const char* key);

    /*!
     * prints the help menu in case the options are not correct.
     */
    virtual void printHelp();

protected:
    /*!
     * Retrieve command switch
     */
    CmdSwitch* getCmdSwitch(const char* key);

    bool token_to_fullkeyname(const string& token, string& fullkey);


private:
    map<string, CmdSwitch*> m_mapKeySwitch;
    map<string, string> m_mapShortcutKeys;
    vector<CmdSwitch*> m_vSwitches;
    string m_strDefaultKey;
    string m_appname;
};

//bool starts_with(const string& src, const string& sub);

}
}
#endif /* CMDLINEPARSER_H_ */
