/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include <assert.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "cmdlineparser.h"
#include "logger.h"

namespace sda {
namespace utils {

bool is_file(const std::string& name) {
    ifstream f(name.c_str());
    if (f.good()) {
        f.close();
        return true;
    } else {
        f.close();
        return false;
    }
}

bool is_number(const std::string& s) {
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

bool starts_with(const string& src, const string& sub) {
    return (src.find(sub) == 0);
}

CmdLineParser::CmdLineParser() {
    // TODO Auto-generated constructor stub
    m_strDefaultKey = "";
    m_appname = "application.exe";
    addSwitch("--help", "-h", "Print Help Options", "");
}

/*
CmdLineParser::CmdLineParser(int argc, char* argv[]) {
        // TODO Auto-generated constructor stub
        assert(parse(argc, argv) > 0);

}
*/

CmdLineParser::~CmdLineParser() {
    // TODO Auto-generated destructor stub
    for (size_t i = 0; i < m_vSwitches.size(); i++) {
        delete m_vSwitches[i];
        m_vSwitches[i] = NULL;
    }

    m_vSwitches.resize(0);
}

bool CmdLineParser::addSwitch(const CmdSwitch& s) {
    CmdSwitch cmd = s;

    if (cmd.desc.length() == 0) {
        LogError("No description provided!");
        return false;
    }

    // check input
    if (cmd.key.find("--") != 0 || cmd.key.length() < 3) {
        LogError("The input key is invalid. Please start with -- and keep a length >= 3");
        return false;
    }

    if (m_mapKeySwitch.find(cmd.key) != m_mapKeySwitch.end()) {
        LogError("This key %s is taken already!", cmd.key.c_str());
        return false;
    }

    if (cmd.shortcut.length() == 0) {
        string temp = "-" + cmd.key[2];

        int i = 3;
        while (m_mapShortcutKeys.find(temp) != m_mapShortcutKeys.end() && (size_t)i < cmd.key.length()) {
            temp = "-" + s.key[i];
            i++;
        }

        cmd.shortcut = temp;
        LogInfo("Automatic shortcut assigned %s to %s", temp.c_str(), cmd.key.c_str());
    }

    if (s.istoggle) {
        cmd.default_value = string("false");
        cmd.value = cmd.default_value;
        cmd.isvalid = true;
    } else {
        cmd.value = cmd.default_value;
        cmd.isvalid = false;
    }

    // add
    CmdSwitch* pcmd = new CmdSwitch(cmd);
    m_vSwitches.push_back(pcmd);
    m_mapShortcutKeys[s.shortcut] = cmd.key;
    m_mapKeySwitch[s.key] = pcmd;

    return true;
}

bool CmdLineParser::addSwitch(
    const string& name, const string& shortcut, const string& desc, const string& default_value, bool istoggle) {
    CmdSwitch s;
    s.key = name;
    s.shortcut = shortcut;
    s.desc = desc;
    s.default_value = default_value;
    s.istoggle = istoggle;

    return addSwitch(s);
}

bool CmdLineParser::setDefaultKey(const char* key) {
    string strKey(key);
    if (!starts_with(strKey, "--")) strKey = "--" + strKey;

    if (m_mapKeySwitch.find(strKey) != m_mapKeySwitch.end()) {
        CmdSwitch* pcmd = m_mapKeySwitch[m_strDefaultKey];
        if (pcmd != NULL) {
            if (pcmd->istoggle) {
                LogError("Boolean command line options can not be used as default keys");
                return false;
            }
        }

        // set default key
        m_strDefaultKey = strKey;
        return true;
    } else
        return false;
}

int CmdLineParser::parse(int argc, char* argv[]) {
    // capture real app name
    if (argc > 0) {
        m_appname = string(argv[0]);
    }

    int i = 0;
    int ctOptions = 0;
    while (i < argc) {
        string key, val;
        bool iskey = false;
        string token = string(argv[i]);

        bool isNextTokenKey = false;
        if (i + 1 < argc) {
            string peeknext = string(argv[i + 1]);
            if (starts_with(peeknext, "-") || starts_with(peeknext, "--")) {
                string fullkey;
                isNextTokenKey = token_to_fullkeyname(peeknext, fullkey);
            }
        }

        // full-key
        if (starts_with(token, string("--"))) {
            if (m_mapKeySwitch.find(token) == m_mapKeySwitch.end()) {
                LogError("Unrecognized key passed %s", token.c_str());
                printHelp();
                return -1;
            }

            key = token;
            iskey = true;
        }
        // shortcut
        else if (starts_with(token, "-")) {
            if (m_mapShortcutKeys.find(token) == m_mapShortcutKeys.end()) {
                LogError("Unrecognized shortcut key passed %s", token.c_str());
                printHelp();
                return -1;
            }

            key = m_mapShortcutKeys[token];
            iskey = true;
        }
        // default key, the value for default key is the last argument
        else if (isNextTokenKey == false && m_strDefaultKey.length() > 0 && i == argc - 2) {
            if (m_mapKeySwitch.find(m_strDefaultKey) == m_mapKeySwitch.end()) {
                LogError("Unrecognized default key %s", m_strDefaultKey.c_str());
                printHelp();
                return -1;
            }

            LogInfo("Using default key: %s", m_strDefaultKey.c_str());
            key = m_strDefaultKey;
            iskey = true;
        }

        // if iskey and needs param then read it
        if (iskey) {
            ctOptions++;

            if (key == "--help") {
                // printf("Reached help prints \n");
                printHelp();
                exit(EXIT_FAILURE);
                return 1;
            }

            // fetch value
            CmdSwitch* pcmd = m_mapKeySwitch[key];

            // read next
            if (pcmd->istoggle) {
                pcmd->value = string("true");
                pcmd->isvalid = true;
            } else {
                i++;
                if (argc <= i) {
                    LogError("Incomplete Switch %s", token.c_str());
                    printHelp();
                    exit(EXIT_FAILURE);
                }
                pcmd->value = string(argv[i]);
                pcmd->isvalid = true;
            }
        }

        // next token
        i++;
    }

    return ctOptions;
}

bool CmdLineParser::token_to_fullkeyname(const string& token, string& fullkey) {
    fullkey = "";
    int ctDashes = 0;
    if (starts_with(token, string("--")))
        ctDashes = 2;
    else if (starts_with(token, string("-")))
        ctDashes = 1;

    if (ctDashes == 0) return false;

    if (ctDashes == 2) {
        if (m_mapKeySwitch.find(token) == m_mapKeySwitch.end()) {
            LogError("Unrecognized key passed %s", token.c_str());
            return false;
        }
        fullkey = token;
    } else if (ctDashes == 1) {
        if (m_mapShortcutKeys.find(token) == m_mapShortcutKeys.end()) {
            LogError("Unrecognized shortcut key passed %s", token.c_str());
            return false;
        }

        fullkey = m_mapShortcutKeys[token];
    }

    return (fullkey.length() > 0);
}

string CmdLineParser::value(const char* key) {
    string strKey(key);
    if (!starts_with(strKey, "--")) strKey = "--" + strKey;

    if (m_mapKeySwitch.find(strKey) != m_mapKeySwitch.end())
        return m_mapKeySwitch[strKey]->value;
    else {
        LogWarn("The input key %s is not recognized!", strKey.c_str());
        return string("");
    }
}

int CmdLineParser::value_to_int(const char* key) {
    string strVal = value(key);
    if (strVal.length() == 0 || !is_number(strVal)) return -1;
    return atoi(strVal.c_str());
}

double CmdLineParser::value_to_double(const char* key) {
    string strVal = value(key);
    if (strVal.length() == 0) return -1;
    return atof(strVal.c_str());
}

bool CmdLineParser::isValid(const char* key) {
    string strKey(key);
    if (!starts_with(strKey, "--")) strKey = "--" + strKey;

    if (m_mapKeySwitch.find(strKey) != m_mapKeySwitch.end())
        return m_mapKeySwitch[strKey]->isvalid;
    else {
        LogWarn("The input key %s is not recognized!", strKey.c_str());
        return false;
    }
}

void CmdLineParser::printHelp() {
    std::cout << "\n==================================================================\n";
    string strAllShortcuts = "";
    for (auto pcmd : m_vSwitches) {
        if (pcmd && !pcmd->shortcut.empty()) strAllShortcuts = strAllShortcuts + pcmd->shortcut;
    }
    // example
    std::cout << "Usage: " << m_appname << " [Options] "
              << "[Files] \n\n";

    // finding maximum
    size_t first_max = 0;
    size_t second_max = 0;
    size_t third_max = 0;

    for (auto pcmd : m_vSwitches) {
        if (pcmd->key.size() > first_max) first_max = pcmd->key.size();
        if (pcmd->shortcut.size() > second_max) second_max = pcmd->shortcut.size();
        if (pcmd->desc.size() > third_max) third_max = pcmd->desc.size();
    }

    // row by row
    for (auto pcmd : m_vSwitches) {
        int no_of_spaces_first = first_max - pcmd->key.size() + 5;
        int no_of_spaces_second = second_max - pcmd->shortcut.size() + 5;

        std::cout << string(10, ' ') << pcmd->key << "," << string(no_of_spaces_first, ' ') << pcmd->shortcut
                  << string(no_of_spaces_second, ' ') << pcmd->desc;

        if (!pcmd->default_value.empty()) {
            int no_of_spaces_third = third_max - pcmd->desc.size() + 5;

            std::cout << string(no_of_spaces_third, ' ') << "Default: [" << pcmd->default_value << "]";
        }

        std::cout << std::endl;
    }
}

CmdLineParser::CmdSwitch* CmdLineParser::getCmdSwitch(const char* key) {
    string strKey(key);
    if (!starts_with(strKey, "--")) strKey = "--" + strKey;

    if (m_mapKeySwitch.find(strKey) != m_mapKeySwitch.end())
        return m_mapKeySwitch[strKey];
    else
        return NULL;
}
}
}
