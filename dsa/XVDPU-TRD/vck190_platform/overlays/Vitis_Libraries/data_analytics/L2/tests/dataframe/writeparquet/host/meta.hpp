// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <utility>

#include "parquet/types.h"
#include "arrow/io/file.h"
#include "parquet/exception.h"
#include "parquet/stream_reader.h"
#include "parquet/stream_writer.h"
#include "parquet/metadata.h"
#include "parquet/types.h"
#include "parquet/thrift_internal.h"
#include "parquet/internal_file_encryptor.h"
#include <fstream>

// This file gives an example of how to use the parquet::StreamWriter
// and parquetReader classes.
// It shows writing/reading of the supported types as well as how a
// user-defined type can be handled.

template <typename T>
using optional = parquet::StreamReader::optional<T>;

// Example of a user-defined type to be written to/read from Parquet
// using C++ input/output operators.
class UserTimestamp {
   public:
    UserTimestamp() = default;

    UserTimestamp(const std::chrono::microseconds v) : ts_{v} {}

    bool operator==(const UserTimestamp& x) const { return ts_ == x.ts_; }

    void dump(std::ostream& os) const {
        const auto t = static_cast<std::time_t>(std::chrono::duration_cast<std::chrono::seconds>(ts_).count());
        os << std::put_time(std::gmtime(&t), "%Y%m%d-%H%M%S");
    }

    void dump(parquet::StreamWriter& os) const { os << ts_; }

   private:
    std::chrono::microseconds ts_;
};

std::ostream& operator<<(std::ostream& os, const UserTimestamp& v) {
    v.dump(os);
    return os;
}

parquet::StreamWriter& operator<<(parquet::StreamWriter& os, const UserTimestamp& v) {
    v.dump(os);
    return os;
}

parquet::StreamReader& operator>>(parquet::StreamReader& os, UserTimestamp& v) {
    std::chrono::microseconds ts;

    os >> ts;
    v = UserTimestamp{ts};

    return os;
}

std::shared_ptr<parquet::schema::GroupNode> GetSchema() {
    parquet::schema::NodeVector fields;

    fields.push_back(parquet::schema::PrimitiveNode::Make("uint64_field", parquet::Repetition::REQUIRED,
                                                          parquet::Type::INT64, parquet::ConvertedType::UINT_64));
    fields.push_back(parquet::schema::PrimitiveNode::Make("uint64_field", parquet::Repetition::REQUIRED,
                                                          parquet::Type::INT64, parquet::ConvertedType::UINT_64));
    //  fields.push_back(parquet::schema::PrimitiveNode::Make("int32_field", parquet::Repetition::REQUIRED,
    //                                                        parquet::Type::INT32, parquet::ConvertedType::INT_32));
    fields.push_back(parquet::schema::PrimitiveNode::Make("uint64_field", parquet::Repetition::REQUIRED,
                                                          parquet::Type::INT64, parquet::ConvertedType::UINT_64));
    fields.push_back(parquet::schema::PrimitiveNode::Make("uint64_field", parquet::Repetition::REQUIRED,
                                                          parquet::Type::INT64, parquet::ConvertedType::UINT_64));
    /*
    fields.push_back(parquet::schema::PrimitiveNode::Make("double_field", parquet::Repetition::REQUIRED,
                                                          parquet::Type::DOUBLE, parquet::ConvertedType::NONE));
    fields.push_back(parquet::schema::PrimitiveNode::Make("double_field", parquet::Repetition::REQUIRED,
                                                          parquet::Type::DOUBLE, parquet::ConvertedType::NONE));
    */
    fields.push_back(parquet::schema::PrimitiveNode::Make("string_field", parquet::Repetition::REQUIRED,
                                                          parquet::Type::BYTE_ARRAY, parquet::ConvertedType::UTF8));

    return std::static_pointer_cast<parquet::schema::GroupNode>(
        parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));
}

struct TestData {
    static const int num_rows = 2000;

    static void init() { std::time(&ts_offset_); }

    // static std::string GetString(const int i) { return "Str #" + std::to_string(i); }
    static std::string GetString(const int i) {
        char candidates[16] = {'a', 'b', 'c', 'd', 'e', 'f', '0', '1', '2', '3', '4', '5', '@', '$', '&', '#'};
        char arry[4];
        arry[0] = candidates[i % 16];
        arry[1] = candidates[(i + 4) % 16];
        arry[2] = candidates[(i + 8) % 16];
        arry[3] = candidates[(i + 12) % 16];
        return std::string(arry);
    }
    static uint64_t GetUInt64(const int i) { return i; }
    static double GetDouble(const int i) { return i; }

    static uint16_t GetUInt16(const int i) { return static_cast<uint16_t>(i); }
    static int32_t GetInt32(const int i) { return 3 * i - 17; }
    static UserTimestamp GetUserTimestamp(const int i) {
        return UserTimestamp{std::chrono::microseconds{(ts_offset_ + 3 * i) * 1000000 + i}};
    }
    static std::chrono::milliseconds GetChronoMilliseconds(const int i) {
        return std::chrono::milliseconds{(ts_offset_ + 3 * i) * 1000ull + i};
    }

    static char char4_array[4];

   private:
    static std::time_t ts_offset_;
    static std::string string_;
};
char TestData::char4_array[] = "XYZ";
std::time_t TestData::ts_offset_;
std::string TestData::string_;
// namespace parquet {
std::unique_ptr<parquet::ThriftSerializer> thrift_serializer_;
void createPage(uint8_t* vl_buffer,
                int vl_sz,
                int vl_nm, //
                uint8_t* chunkbuffer,
                int& chunk_sz,
                int& chunk_nm,
                int& num_pages) {
    thrift_serializer_.reset(new parquet::ThriftSerializer);
    parquet::format::PageHeader page_header;
    page_header.__set_uncompressed_page_size(static_cast<int32_t>(vl_sz));
    page_header.__set_compressed_page_size(static_cast<int32_t>(vl_sz));
    parquet::format::DataPageHeader data_page_header;
    data_page_header.__set_num_values(vl_nm);
    data_page_header.__set_encoding(ToThrift(parquet::Encoding::PLAIN));
    data_page_header.__set_definition_level_encoding(ToThrift(parquet::Encoding::RLE));
    data_page_header.__set_repetition_level_encoding(ToThrift(parquet::Encoding::RLE));
    page_header.__set_type(parquet::format::PageType::DATA_PAGE);
    page_header.__set_data_page_header(data_page_header);
    uint32_t header_sz;
    uint8_t* header_buf;
    thrift_serializer_->SerializeToBuffer(&page_header, &header_sz, &header_buf);
    memcpy(chunkbuffer + chunk_sz, header_buf, header_sz);
    chunk_sz += header_sz;

    memcpy(chunkbuffer + chunk_sz, vl_buffer, vl_sz);
    chunk_sz += vl_sz;
    chunk_nm += vl_nm;

    num_pages++;
}
void createChunkColumnMeta(parquet::ColumnChunkMetaDataBuilder* metadata_,
                           int64_t num_values,
                           int data_page_offset,
                           int page_ordinal,
                           uint8_t* columnbuffer,
                           int& columnbuffer_sz,
                           int column_id) {
    int total_uncompressed_size = columnbuffer_sz;

    std::shared_ptr<parquet::Encryptor> meta_encryptor_ = nullptr;
    std::map<parquet::Encoding::type, int32_t> dict_encoding_stats;
    std::map<parquet::Encoding::type, int32_t> data_encoding_stats;
    data_encoding_stats[parquet::Encoding::PLAIN] = page_ordinal;
    std::cout << "DEBUG: createChunkMeta"
              << "num_values:" << num_values << ", data_page_offset:" << data_page_offset
              << ",total_uncompressed_size:" << total_uncompressed_size << std::endl;
    metadata_->Finish(num_values, 0, -1, data_page_offset, total_uncompressed_size, total_uncompressed_size, false,
                      true, dict_encoding_stats, data_encoding_stats, meta_encryptor_);
    uint32_t meta_size = 0;
    thrift_serializer_->SerializeToBuffer((parquet::format::ColumnChunk*)metadata_->contents(), &meta_size,
                                          &columnbuffer);
    // columnbuffer_sz += meta_size;
}
template <int W>
void WriteParquetFile(ap_uint<64>* buff0, int colum_num) {
    std::cout << "Header size:" << buff0[0] << std::endl;
    ap_uint<64>* s_buff = buff0 + 1;
    ap_uint<64>* l_buff = buff0 + 1 + 16;
    ap_uint<64>* f_buff = buff0 + 1 + 16 + 16;
    ap_uint<64>* nodes = buff0 + 1 + 16 + 16 + 16;
    ap_uint<64>* pinfos[1 << W];
    // = buff0 + 1 + 16 + 16 + 16 + 1024;
    for (int i = 0; i < colum_num; i++) {
        pinfos[i] = buff0 + 1 + 16 + 16 + 16 + 1024 + 1024 * i;
        for (int j = 0; j < 10; j++) {
            int page_sz = pinfos[i][j].range(63, 32);
            int page_nm = pinfos[i][j].range(31, 0);
            std::cout << "Field " << i << ", page num: " << page_nm << ", page size: " << page_sz << std::endl;
        }
    }

    parquet::WriterProperties::Builder builder;
    builder.disable_dictionary();
    std::shared_ptr<parquet::KeyValueMetadata> key_value_metadata_ = nullptr;
    parquet::SchemaDescriptor descriptor;
    descriptor.Init(GetSchema());
    auto metadata_ = parquet::FileMetaDataBuilder::Make(&descriptor, builder.build(), key_value_metadata_);

    // std::vector<parquet::RowGroupMetaDataBuilder*> rg_metadatas;
    std::ofstream parquet_file;
    parquet_file.open("parquet-stream-api-example_xil.parquet");
    // write header
    const char kParquetMagic[4] = {'P', 'A', 'R', '1'};
    parquet_file.write(kParquetMagic, 4);

    int num_pages[1 << W];

    /* chunck column
    *  -- header buffer
    *  -- DL     buffer
    *  -- data   buffer
    *  -- header buffer
    *  -- DL     buffer
    *  -- data   buffer
    *  -- header buffer
    *  -- DL     buffer
    *  -- data   buffer
    */
    uint8_t* columnbuffer[1 << W];
    int columnbuffer_sz[1 << W];
    int columnbuffer_len[1 << W];

    int data_page_offset = 4;
    for (int i = 0; i < 10; i++) {
        columnbuffer[i] = (uint8_t*)malloc(1 << 30);
        columnbuffer_sz[i] = 0;
        columnbuffer_len[i] = 0;
        num_pages[i] = 0;
    }
    // now only support one row group
    for (int grp_id = 0; grp_id < 1; grp_id++) {
        auto rg_metadata_ = metadata_->AppendRowGroup();
        // int grp_row_num = group_rows[grp_id];
        // create a page
        int total_bytes_written_ = 0;
        int data_page_offset_0 = data_page_offset;
        for (int j = 0; j < colum_num; j++) {
            ap_uint<32> header_node_id = f_buff[j].range(11, 0);
            ap_uint<32> tail_node_id = f_buff[j].range(23, 12);
            ap_uint<64>* page_info = pinfos[j];

            ap_uint<32> node_id = header_node_id;
            int offset = 0;
            int page_counter = 0;
            std::cout << "header: " << header_node_id << ", tail: " << tail_node_id << std::endl;
            while (node_id != tail_node_id) {
                int page_sz = page_info[page_counter].range(63, 32);
                int page_nm = page_info[page_counter].range(31, 0);
                // get the addr in buffer
                ap_uint<32> addr = 0;
                addr.range(31, 20) = nodes[node_id].range(23, 12);
                std::cout << "Create Page: "
                          << "page sz:" << page_sz << ", page nm:" << page_nm << std::endl;
                createPage(reinterpret_cast<uint8_t*>(buff0 + addr), page_sz, page_nm, columnbuffer[j],
                           columnbuffer_sz[j], columnbuffer_len[j], num_pages[j]);

                node_id = nodes[node_id].range(11, 0);
                page_counter++;
            }
            int page_sz = page_info[page_counter].range(63, 32);
            int page_nm = page_info[page_counter].range(31, 0);
            // get the addr in buffer
            ap_uint<32> addr = 0;
            addr.range(31, 20) = nodes[node_id].range(23, 12);
            std::cout << "Create Page: "
                      << "page sz:" << page_sz << ", page nm:" << page_nm << std::endl;
            createPage(reinterpret_cast<uint8_t*>(buff0 + addr), page_sz, page_nm, columnbuffer[j], columnbuffer_sz[j],
                       columnbuffer_len[j], num_pages[j]);

            node_id = nodes[node_id].range(11, 0);
            page_counter++;
            // update
            std::cout << "Create ColumnChunK: "
                      << "column sz:" << columnbuffer_sz[j] << ",column len:" << columnbuffer_len[j] << std::endl;

            auto metadata_ = rg_metadata_->NextColumnChunk();
            createChunkColumnMeta(metadata_, columnbuffer_len[j], data_page_offset, num_pages[j], columnbuffer[j],
                                  columnbuffer_sz[j], j);
            total_bytes_written_ += columnbuffer_sz[j];
            // std::cout << "total_bytes_written_: " << total_bytes_written_ << std::endl;
            parquet_file.write(reinterpret_cast<char*>(columnbuffer[j]), columnbuffer_sz[j]);
            data_page_offset += columnbuffer_sz[j];
            num_pages[j] = 0;
            columnbuffer_len[j] = 0;
            columnbuffer_sz[j] = 0;
        }

        // create chunkcolumndata
        // RowGroupMetaDataBuilder* metadata_;
        // metadata_->set_num_rows(num_rows_);
        // metadata_->Finish(total_bytes_written_, row_group_ordinal_);
        std::cout << std::endl;
        rg_metadata_->set_num_rows(grp_id);
        rg_metadata_->Finish(total_bytes_written_, grp_id);
    }
    // WriteFileMetaData
    // CreateFooter(metadata_);
    // std::unique_ptr<format::FileMetaData> metadata_;
    std::shared_ptr<parquet::FileMetaData> file_metadata_ = metadata_->Finish();
    // write data
    // int meta_len = file_metadata.WriteTo(sink);
    // serializer.Serialize(metadata_.get(), dst, encryptor);
    // uint8_t* out_buffer = (uint8_t*)malloc(1024);
    uint32_t metadata_len = 0;
    // SerializeToBuffer(file_metadata_.get(), &metadata_len, &out_buffer);
    std::string metadata = file_metadata_->SerializeToString();
    metadata_len = metadata.length();
    parquet_file.write(metadata.c_str(), metadata_len);

    std::cout << "DEBUG: metadata_len: " << metadata_len << std::endl;
    parquet_file.write(reinterpret_cast<char*>(&metadata_len), 4);
    parquet_file.write(kParquetMagic, 4);
    std::cout << "Parquet Stream Writing complete." << std::endl;
}

int ReadParquetFile() {
    int nerr = 0;
    std::shared_ptr<arrow::io::ReadableFile> infile;

    PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open("parquet-stream-api-example_xil.parquet"));

    parquet::StreamReader os{parquet::ParquetFileReader::Open(infile)};
    std::cout << "-------------------------------------------------\n";
    // std::cout << os.num_columns() << ", " << os.num_rows() << std::endl;

    uint64_t uint64_0;
    uint64_t uint64_1;
    uint64_t d_2;
    uint64_t d_3;
    std::string str_4;
    int i;

    for (i = 0; !os.eof(); ++i) {
        os >> uint64_0;
        os >> uint64_1;
        os >> d_2;
        os >> d_3;
        os >> str_4;
        os >> parquet::EndRow;
        // std::cout << str_4 << ":" << TestData::GetString(i) << std::endl;
        /*
        if (0) {
            // For debugging.
            std::cout << "Row #" << i << std::endl;

            std::cout << "string[";
            std::cout << req_string;
            std::cout << "] int32[" << int32;
            std::cout << "] uint64[";
            std::cout << uint64;
            std::cout << "] double[" << d << "]" << std::endl;
        }
        */
        // Check data.
        if (uint64_0 != i + 0) nerr++;
        if (uint64_1 != i + 1) nerr++;
        if (d_2 != i + 2) nerr++;
        if (d_3 != i + 3) nerr++;
        if (str_4.substr(0, 4) != TestData::GetString(i).substr(0, 4)) nerr++;
        // assert(std::labs(d_2 - (i + 2)) > 1e-6);
        // assert(std::labs(d_3 - (i + 2)) > 1e-6);
        // assert(ts_user == TestData::GetUserTimestamp(i));
        // assert(ts_ms == TestData::GetChronoMilliseconds(i));
    }
    // assert(TestData::num_rows == i);

    std::cout << "Parquet Stream Reading complete." << std::endl;
    return nerr;
}
