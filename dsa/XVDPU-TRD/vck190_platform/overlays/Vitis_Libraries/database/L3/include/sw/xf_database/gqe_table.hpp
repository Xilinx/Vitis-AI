/*
 * Copyright 2020 Xilinx, Inc.
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
#ifndef _GQE_TABLE_L3_
#define _GQE_TABLE_L3_
// commmon
#include <iostream>
#include <vector>
#include <algorithm>

namespace xf {
namespace database {
namespace gqe {

enum ErrCode { SUCCESS = 0, PARAM_ERROR = 1, MEM_ERROR = 2, DEV_ERROR = 3 };
enum class TypeEnum { TypeInt32 = sizeof(int32_t), TypeInt64 = sizeof(int64_t), TypeBool };

struct ColPtr {
    void* ptr;
    int len;
};

/**
 * @brief handling column-related works (name, section, pointer, etc.)
 *
 */

class Table {
   private:
    std::vector<std::string> _col_name;
    std::vector<std::vector<char*> > _col_ptr;
    std::vector<char*> _val_ptr;
    std::vector<int> _col_type_size;
    std::vector<int> _sec_nrow;
    int _valid_col_num;
    int _nrow;
    int _sec_num;
    std::string _tab_name;

    bool _rowID_en_flag;
    bool _valid_en_flag;

    std::string _rowID_col_name;
    std::string _valid_col_name;

   public:
    /**
     * @brief construct of Table.
     *
     * @param _name name of the table.
     *
     */
    Table(std::string _name);

    /**
     * @brief add one column into the Table with user-provided buffer pointer.
     *
     * usage:
     * tab.addCol("o_orderkey", TypeEnum::TypeInt32, tab_o_col0, 10000);
     *
     * @param _name column name
     * @param type_size size of the column element data type in bytes
     * @param _ptr user-provided column buffer pointer
     * @param row_num number of rows
     *
     */
    void addCol(std::string _name, TypeEnum type_size, void* _ptr, int row_num);
    /**
     * @brief allocate buffer for one column and add it into the Table.
     *
     * usage:
     * tab.addCol("o_orderkey", TypeEnum::TypeInt32, 10000);
     *
     * @param _name column name
     * @param type_size size of the column element data type in bytes
     * @param row_num number of rows
     *
     */
    void addCol(std::string _name, TypeEnum type_size, int row_num);
    /**
     * @brief create one column with several sections by loading rows from data file.
     *
     * usage:
     * tab.addCol("o_orderkey", TypeEnum::TypeInt32, {file1.dat,file2.dat});
     *
     * @param _name column name
     * @param type_size size of the column element data type in bytes
     * @param dat_list data file list
     *
     */
    void addCol(std::string _name, TypeEnum type_size, std::vector<std::string> dat_list);
    /**
     * @brief create one column with several sections by user-provided pointer list
     *
     * usage:
     * tab.addCol("o_orderkey", TypeEnum::TypeInt32, {{ptr1,10000},{ptr2,20000}});
     *
     * @param _name column name
     * @param type_size size of the column element data type in bytes
     * @param ptr_pairs vector of (ptr,row_num) pairs
     *
     */
    void addCol(std::string _name, TypeEnum type_size, std::vector<struct ColPtr> ptr_pairs);

    /**
     * @brief add validation column with user-provided validation pointer list
     *
     * **Caution** This is an experimental-only API, will be deprecated in the next release.
     *
     * @param _rowid_name name of row-id column
     * @param _valid_name name of validation bits column
     * @param _rowid_en enable flag of row-id
     * @param _valid_en enable flag of validation bits
     * @param validationPtrVector validation bits pointer list
     *
     */
    void genRowIDWithValidation(std::string _rowid_name,
                                std::string _valid_name,
                                bool _rowid_en,
                                bool _valid_en,
                                std::vector<char*> validationPtrVector);

    /**
     * @brief add validation column with user-provided pointer
     *
     * **Caution** This is an experimental-only API, will be deprecated in the next release.
     *
     * @param _rowid_name name of row-id column
     * @param _valid_name name of validation bits column
     * @param _rowid_en enable flag of row-id
     * @param _valid_en enable flag of validation bits
     * @param ptr validation bits column pointer
     * @param row_num number of rows
     *
     */
    void genRowIDWithValidation(
        std::string _rowid_name, std::string _valid_name, bool _rowid_en, bool _valid_en, void* ptr, int row_num);

    /**
     * @brief add validation column with user-provided data file list
     *
     * **Caution** This is an experimental-only API, will be deprecated in the next release.
     *
     * @param _rowid_name name of row-id column
     * @param _valid_name name of validation bits column
     * @param _rowid_en enable flag of row-id
     * @param _valid_en enable flag of validation bits
     * @param dat_list data file list
     *
     */
    void genRowIDWithValidation(std::string _rowid_name,
                                std::string _valid_name,
                                bool _rowid_en,
                                bool _valid_en,
                                std::vector<std::string> dat_list);

    /**
     * @brief set number of rows for the entire table.
     *
     * @param _num number of rows of the entire table
     *
     */
    void setRowNum(int _num);

    /**
     * @brief get number of rows of the entire table.
     * @return number of rows of the entire table
     */
    size_t getRowNum() const;
    /**
     * @brief get number of rows for the specified section.
     *
     * @param sid section ID
     * @return number of rows of the specified section
     */
    size_t getSecRowNum(int sid) const;

    /**
     * @brief get number of columns.
     * @return number of columns of the table.
     */
    size_t getColNum() const;
    /**
     * @brief get number of sections.
     * @return number of sections of the table.
     */
    size_t getSecNum() const;

    /**
     * @brief divide the columns evenly if section number is greater than 0.
     * @param sec_l number of sections, if 0, do nothing since everything is done by addCol with json input.
     */
    void checkSecNum(int sec_l);

    /**
     * @brief get column data type size.
     * @param cid column ID
     * @return date type size of input column id.
     */
    size_t getColTypeSize(int cid) const;

    /**
     * @brief get buffer pointer.
     *
     * when getColPointer(2,4,1), it means the 2nd column was divied into 4 sections,
     * return the pointer of the 2nd section
     *
     * @param i column id
     * @param _slice_num divide column i into _slice_num parts
     * @param j get the j'th part pointer after dividing
     *
     * @return column buffer pointer
     *
     */
    char* getColPointer(int i, int _slice_num, int j = 0) const;

    /**
     * @brief get the validation buffer pointer
     *
     * @param _slice_num number of sections of the validation column
     * @param j the index of the section
     * @return the pointer of the specified section
     *
     */
    char* getValColPointer(int _slice_num, int j) const;

    /**
     * @brief get column pointer.
     *
     * @param i column id
     * @return column pointer
     *
     */
    char* getColPointer(int i) const;

    /**
     * @brief set col_names
     *
     * @param col_name column name list
     *
     */
    void setColNames(std::vector<std::string> col_names);

    /**
     * @brief get col_names
     * @return list of column names
     */
    std::vector<std::string> getColNames();

    /**
     * @brief get the name of the row-id column
     */
    std::string getRowIDColName();

    /**
     * @brief get the name of the validation bits column
     */
    std::string getValidColName();

    /**
     * @brief get row-id enable flag
     */
    bool getRowIDEnableFlag();

    /**
     * @brief get validation bits enable flag
     */
    bool getValidEnableFlag();

    /**
     * @brief deconstructor of Table.
     *
     */
    ~Table();

    /**
     * @brief print information of the table
     */
    void info();
};
}
}
}
#endif
