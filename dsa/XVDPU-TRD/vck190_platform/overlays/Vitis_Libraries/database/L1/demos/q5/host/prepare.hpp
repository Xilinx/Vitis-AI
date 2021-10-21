#ifndef PREPARE_HPP
#define PREPARE_HPP

#include "utils.hpp"
#include "tpch_read_3.hpp"

#include <string>
#include <iostream>
#include <cstdlib>

namespace {

void generate_tbl(const std::string& dbgen_dir, const std::string& tbl_dir, const int sf) {
    std::string cmd = "git clone https://github.com/electrum/ssb-dbgen.git " + dbgen_dir;
    system(cmd.c_str());
    cmd = "sed -i 's/MACHINE =MAC/MACHINE =LINUX/' " + dbgen_dir + "/makefile";
    system(cmd.c_str());
    cmd = "sed -i 's/WORKLOAD =SSBM/WORKLOAD =TPCH/' " + dbgen_dir + "/makefile";
    system(cmd.c_str());
    cmd = "make -C " + dbgen_dir + " 1>" + dbgen_dir + "/mk.log 2>&1";
    system(cmd.c_str());
    cmd = "cd " + dbgen_dir + " && ./dbgen -f -s " + std::to_string(sf);
    system(cmd.c_str());
    cmd = "mkdir -p " + tbl_dir;
    system(cmd.c_str());
    cmd = "mv " + dbgen_dir + "/*.tbl " + tbl_dir;
    system(cmd.c_str());
    cmd = "rm -rf " + dbgen_dir;
    system(cmd.c_str());
}

void convert_dat(const std::string& in_dir, const std::string& out_dir) {
    std::string cmd = "mkdir -p " + out_dir;
    system(cmd.c_str());

    std::string r_path = in_dir + "/region.tbl";
    if (!is_file(r_path)) {
        printf("ERROR: \"%s\" is not a file!\n", r_path.c_str());
    }
    std::string n_path = in_dir + "/nation.tbl";
    if (!is_file(n_path)) {
        printf("ERROR: \"%s\" is not a file!\n", n_path.c_str());
    }
    std::string c_path = in_dir + "/customer.tbl";
    if (!is_file(c_path)) {
        printf("ERROR: \"%s\" is not a file!\n", c_path.c_str());
    }
    std::string o_path = in_dir + "/orders.tbl";
    if (!is_file(o_path)) {
        printf("ERROR: \"%s\" is not a file!\n", o_path.c_str());
    }
    std::string l_path = in_dir + "/lineitem.tbl";
    if (!is_file(l_path)) {
        printf("ERROR: \"%s\" is not a file!\n", l_path.c_str());
    }
    std::string s_path = in_dir + "/supplier.tbl";
    if (!is_file(s_path)) {
        printf("ERROR: \"%s\" is not a file!\n", s_path.c_str());
    }

    struct timeval tv0, tv1;
    gettimeofday(&tv0, 0);

    extract_col<region_t>(r_path, out_dir, r_regionkey(), r_name());
    extract_col<nation_t>(n_path, out_dir, n_nationkey(), n_regionkey(), n_name());
    extract_col<customer_t>(c_path, out_dir, c_custkey(), c_nationkey());
    extract_col<orders_t>(o_path, out_dir, o_orderkey(), o_custkey(), o_orderdate());
    extract_col<lineitem_t>(l_path, out_dir, l_orderkey(), l_suppkey(), l_extendedprice(), l_discount());
    extract_col<supplier_t>(s_path, out_dir, s_suppkey(), s_nationkey());

    gettimeofday(&tv1, 0);
    int usec = tvdiff(&tv0, &tv1);
    printf("Time to columnize tables: %d usec.\n", usec);
}

std::string prepare(std::string& wrk_dir, const int sf) {
    const bool debug = true;
    std::string dbgen_dir = wrk_dir;
    dbgen_dir += "/dbgen";
    std::string tbl_dir = wrk_dir;
    tbl_dir += "/tbl";
    std::string dat_dir = wrk_dir;
    dat_dir += "/dat" + std::to_string(sf);
    if (debug) std::cout << "dat dir is: " << dat_dir << std::endl;
    if (is_file(dat_dir + "/l_orderkey.dat")) {
        if (debug) std::cout << "file exist, skip generate." << std::endl;
    } else {
        if (debug) std::cout << "file not exist, generate." << std::endl;
        if (!is_dir(wrk_dir)) {
            std::string cmd = "mkdir -p ";
            cmd += wrk_dir;
            system(cmd.c_str());
        }
        generate_tbl(dbgen_dir, tbl_dir, sf);
        convert_dat(tbl_dir, dat_dir);
    }
    return dat_dir;
}
}

#endif
