/*
 * Copyright 2021 Xilinx, Inc.
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

#pragma once

#ifndef __XF_VISION_MESSAGE_H__
#define __XF_VISION_MESSAGE_H__

#include <iostream>
#include <string>
#include <map>

#define XF_APP_SUCCESS 0
#define XF_APP_FAILURE 1

using namespace std;

class XfVisionMessage {
   public:
    enum MessageSeverityType {
        PRINT = 0,
        FATAL = 1,
        ERROR = 2,
        WARNING = 3,
        INFO = 4,
        DEBUG = 7,

        INFO_LOW = 4,
        INFO_MED = 5,
        INFO_HIGH = 6,

        DEBUG_LOW = 7,
        DEBUG_MED = 8,
        DEBUG_HIGH = 9
    };

   private:
    int SeverityLevel;
    int ExitLevel;

    const int c_PrefixMsgWidth = 10;

    map<string, MessageSeverityType> MsgLevel;

   public:
    XfVisionMessage() {
        SeverityLevel = PRINT;
        ExitLevel = ERROR;

        MsgLevel["PRINT"] = PRINT;
        MsgLevel["FATAL"] = FATAL;
        MsgLevel["ERROR"] = ERROR;
        MsgLevel["WARNING"] = WARNING;
        MsgLevel["INFO"] = INFO;
        MsgLevel["DEBUG"] = DEBUG;

        MsgLevel["INFO_LOW"] = INFO_LOW;
        MsgLevel["INFO_MED"] = INFO_MED;
        MsgLevel["INFO_HIGH"] = INFO_HIGH;

        MsgLevel["DEBUG_LOW"] = DEBUG_LOW;
        MsgLevel["DEBUG_MED"] = DEBUG_MED;
        MsgLevel["DEBUG_HIGH"] = DEBUG_HIGH;
    }

    int severityLevel(int lvl = -1) {
        if (lvl >= 0) SeverityLevel = lvl;

        return SeverityLevel;
    }

    int exitLevel(int lvl = -1) {
        if (lvl >= 0) ExitLevel = lvl;

        return ExitLevel;
    }

    int message(string MsgType, string msg, bool IgnoreLevelCheck = false) {
        if ((SeverityLevel >= MsgLevel[MsgType]) || IgnoreLevelCheck)
            cout << "[" << right << setw(c_PrefixMsgWidth) << MsgType << "] " << left << msg << endl;

        return ((ExitLevel > MsgLevel[MsgType]) || IgnoreLevelCheck) ? XF_APP_SUCCESS : XF_APP_FAILURE;
    }

    int print(string msg) { return message("PRINT", msg, true); }

    int fatal(string msg) { return message("FATAL", msg, false); }
    int error(string msg) { return message("ERROR", msg, false); }
    int warning(string msg) { return message("WARNING", msg, false); }
    int info(string msg) { return message("INFO", msg, false); }
    int debug(string msg) { return message("DEBUG", msg, false); }

    int lowInfo(string msg) { return message("INFO_LOW", msg, false); }
    int medInfo(string msg) { return message("INFO_MED", msg, false); }
    int highInfo(string msg) { return message("INFO_HIGH", msg, false); }

    int lowDebug(string msg) { return message("DEBUG_LOW", msg, false); }
    int medDebug(string msg) { return message("DEBUG_MED", msg, false); }
    int highDebug(string msg) { return message("DEBUG_HIGH", msg, false); }
};

#endif // __XF_VISION_MESSAGE_H__
