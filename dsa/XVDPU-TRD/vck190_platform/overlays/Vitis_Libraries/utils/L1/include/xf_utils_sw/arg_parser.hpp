/*
 * Copyright 2019 Xilinx, Inc.
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
 */

#ifndef XF_UTILS_SW_ARGPARSER_HPP
#define XF_UTILS_SW_ARGPARSER_HPP

#include <algorithm>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <cstdlib>
#include <cstdio>
#include <cassert>

namespace xf {
namespace common {
namespace utils_sw {

namespace details {
//  handle string wstring
template <typename T>
T base_name(T path, T const& delims = "/\\") {
    return path.substr(path.find_last_of(delims) + 1);
}

template <typename T>
T to_upper(const T& orig) {
    T s = orig;
    for (auto& c : s) c = std::toupper(c);
    return s;
}
} // details

class ArgParser {
   private:
    void checkStyle(const std::string& opt, const std::string& opt_full, bool is_flag = false) {
        std::string t = is_flag ? "flag" : "option";
        if (opt != "") {
            if (opt.length() > 2) {
                _log << "ERROR: short " << t << " " << opt << " is too long." << std::endl;
                exit(1);
            }
            if (opt[0] != '-') {
                _log << "ERROR: short " << t << " " << opt << " should begin with dash." << std::endl;
                exit(1);
            }
        }
        if (opt_full != "") {
            if (opt_full.length() < 4) {
                _log << "ERROR: long " << t << " " << opt_full << " is too short." << std::endl;
                exit(1);
            }
            if (opt_full[0] != '-' || opt_full[1] != '-') {
                _log << "ERROR: long " << t << " " << opt_full << " should begin with double dash." << std::endl;
                exit(1);
            }
        }
    }

   public:
    ArgParser(int argc, const char* argv[], std::ostream& log = std::cerr) : _log(log) {
        _bin_name = details::base_name(std::string(argv[0]));
        for (int i = 1; i < argc; ++i) _tokens.push_back(std::string(argv[i]));
        addFlag("-h", "--help", "Show usage");
    }
    /// @brief add an boolean flag.
    /// The default value is always 'false'.
    ///
    /// @param opt the short opt name, must start with "-", or being "" if not used.
    /// @param opt_full the full opt name, must start with "--", pass "" if not used.
    /// @param info the info to be shown in usage.
    void addFlag(const std::string opt, const std::string opt_full, const std::string info) {
        checkStyle(opt, opt_full, true);
        if (opt == "") {
            if (opt_full != "") {
                _flags.emplace_back(opt_full, info);
            } else {
                _log << "ERROR: short and full option name cannot be both empty." << std::endl;
                exit(1);
            }
        } else {
            if (opt_full != "") {
                _opt_map.insert({{opt, opt_full}, {opt_full, opt}});
            }
            _flags.emplace_back(opt, info);
        }
    }
    /// @brief add an option with value.
    ///
    /// @param opt the short opt name, must start with "-".
    /// @param opt_full the full opt name, must start with "--", pass "" if not used.
    /// @param info the info to be shown in usage.
    /// @param def the default value string.
    /// @param required if an option is required, program with exit upon failure of obtaining value,
    ///                 otherwise default value will be provide.
    void addOption(const std::string opt,
                   const std::string opt_full,
                   const std::string info,
                   const std::string def,
                   bool required = false) {
        checkStyle(opt, opt_full, false);
        if (opt == "") {
            if (opt_full != "") {
                _opts.emplace_back(opt_full, info, def, required);
            } else {
                _log << "ERROR: flag name cannot be empty." << std::endl;
                exit(1);
            }
        } else {
            if (opt_full != "") {
                _opt_map.insert({{opt, opt_full}, {opt_full, opt}});
            }
            _opts.emplace_back(opt, info, def, required);
        }
    }

    /// @brief get value of option by name without leading dash.
    /// Either short or long option name can be used.
    /// Example: p.getAs<int>("longopt");
    template <typename T>
    T getAs(const std::string name) const {
        std::string dash = "-", opt;
        if (name.size() > 1) {
            opt = dash + dash + name;
        } else {
            opt = dash + name;
        }
        return getAs(opt, type<T>());
    };

    /// @brief print usage to stdout.
    void showUsage() const {
        const int pos = 24;
        printf("Usage: %s [OPTIONS]\n\n", _bin_name.c_str());

        for (auto t : _flags) {
            const std::string& f = std::get<0>(t); // flag
            int n = printf("  %s", f.c_str());
            std::string alt = getAlt(f);
            if (alt != "") {
                n += printf(", %s", alt.c_str());
            }
            if (n >= pos) {
                n = 0;
                printf("\n");
            }
            for (int i = n; i < pos; ++i) {
                printf(" ");
            }
            printf("%s.\n", std::get<1>(t).c_str()); // info
        }
        for (auto t : _opts) {
            const std::string& f = std::get<0>(t); // opt
            int n = printf("  %s", f.c_str());
            std::string alt = getAlt(f);
            if (alt != "") {
                n += printf(", %s %s", alt.c_str(),
                            details::to_upper(details::base_name(alt, std::string("-"))).c_str());
            } else {
                n += printf(" %s", details::to_upper(details::base_name(f, std::string("-"))).c_str());
            }
            if (n >= pos) {
                n = 0;
                printf("\n");
            }
            for (int i = n; i < pos; ++i) {
                printf(" ");
            }
            printf("%s. ", std::get<1>(t).c_str()); // info, def
            if (std::get<3>(t)) {
                // required
                printf("Required.\n");
            } else {
                printf("Default: %s\n", std::get<2>(t).c_str()); // def
            }
        }
    }

   private:
    template <typename T>
    struct type {};

    // bool flag or opt
    bool getAs(const std::string name, type<bool>) const {
        bool is_flag = false;
        bool t = findFlag(name, is_flag);
        if (is_flag) {
            return t;
        }
        std::string s = findOpt(name);
        if (s == "T" || s == "t" || s == "True" || s == "true" || s == "1") {
            return true;
        } else if (s == "F" || s == "f" || s == "False" || s == "false" || s == "0") {
            return false;
        } else if (s != "") {
            _log << "ERROR: parsing option \"" << name << "\"'s value '" << s << "' as bool." << std::endl;
            exit(1);
        }
        return false;
    }
    // int opt
    int getAs(const std::string name, type<int>) const {
        int t = 0;
        std::string s = findOpt(name);
        try {
            t = std::stoi(s);
        } catch (std::exception const& e) {
            _log << "ERROR: parsing option \"" << name << "\"'s value '" << s << "' as int: " << e.what() << "."
                 << std::endl;
            exit(1);
        }
        return t;
    }
    // string opt
    std::string getAs(const std::string name, type<std::string>) const { return findOpt(name); }

    std::ostream& _log;
    std::string _bin_name;
    std::vector<std::string> _tokens;

    // name, info
    std::vector<std::tuple<std::string, std::string> > _flags;
    // name, info, def, required
    std::vector<std::tuple<std::string, std::string, std::string, bool> > _opts;
    // lookup
    std::unordered_map<std::string, std::string> _opt_map;
    std::string getAlt(const std::string& name) const {
        std::string alt = "";
        {
            auto itr = _opt_map.find(name);
            if (itr != _opt_map.end()) {
                alt = itr->second;
            }
        }
        return alt;
    }
    // @param is_flag return whether it is known flag
    // @return existance of flag
    bool findFlag(const std::string flag, bool& is_flag) const {
        std::string alt = getAlt(flag);
        auto t = std::find_if(_flags.begin(), _flags.end(), [&flag, &alt](const auto& a) -> bool {
            return std::get<0>(a) == flag || std::get<0>(a) == alt;
        });
        if (t != _flags.end()) {
            is_flag = true;
            // flag search
            {
                auto itr = std::find(_tokens.begin(), _tokens.end(), flag);
                if (itr != _tokens.end()) {
                    return true;
                }
            }
            // alternative search
            {
                auto itr = std::find(_tokens.begin(), _tokens.end(), alt);
                if (itr != _tokens.end()) {
                    return true;
                }
            }
            return false;
        }
        is_flag = false;
        return false;
    }
    // @return argument string if found or default string if not found
    std::string findOpt(const std::string opt) const {
        // opt search
        {
            auto itr = std::find(_tokens.begin(), _tokens.end(), opt);
            if (itr != _tokens.end()) {
                if (++itr != _tokens.end()) {
                    return *itr;
                }
            }
        }
        // alternative search
        std::string alt = getAlt(opt);
        {
            auto itr = std::find(_tokens.begin(), _tokens.end(), alt);
            if (itr != _tokens.end()) {
                if (++itr != _tokens.end()) {
                    return *itr;
                }
            }
        }
        // get default with opt
        {
            auto itr = std::find_if(_opts.begin(), _opts.end(), [&opt, &alt](const auto& a) -> bool {
                return std::get<0>(a) == opt || std::get<0>(a) == alt;
            });
            if (itr != _opts.end()) {
                if (std::get<3>(*itr)) {
                    _log << "ERROR: option '" << opt << "' is required but not provided. Run with '-h' to see usage."
                         << std::endl;
                    exit(1);
                }
                return std::get<2>(*itr);
            } else {
                _log << "ERROR: unknown option '" << opt << "'." << std::endl;
                exit(1);
            }
        }

        // should never reach here...
        assert(0 && "ArgParser bug?");
        return "";
    }
};

} // utils_sw
} // common
} // xf

#endif // XF_UTILS_SW_ARGPARSER_HPP
