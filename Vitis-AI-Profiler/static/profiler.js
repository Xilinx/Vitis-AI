/*
 * Copyright 2020 Xilinx Inc.
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

var hwInfoData = [];
var util_data = [];
var cg_data = [];
var task_data = [];
var data = [];
var dataCG = [];
var startTime
var endTime
var categories = [];
var APMMap = [
    { port: "DDRC_PORT_S1", description: "APU, PL_S_AXI_HPC0_FPD, PL_S_AXI_HPC1_FPD" },
    { port: "DDRC_PORT_S2", description: "APU, PL_S_AXI_HPC0_FPD, PL_S_AXI_HPC1_FPD" },
    { port: "DDRC_PORT_S3", description: "DisplayPort, PL_S_AXI_HP0_FPD" },
    { port: "DDRC_PORT_S4", description: "PL_S_AXI_HP1_FPD, PL_S_AXI_HP2_FPD" },
    { port: "DDRC_PORT_S5", description: "PL_S_AXI_HP3_FPD, FPD DMA" }
];

var axiTraffic = [];
var FPS = [];
var XIR_LINKS = [];
var XIR_NODES = [];
var XIR_TGT = [];
var vaiProfilerChart = echarts.init(document.getElementById('mainChart'));

vaiProfilerChart.showLoading();

var xhr = new XMLHttpRequest();

xhr.onload = function() {
    if (xhr.status == 200) {
        var dpu_data = JSON.parse(xhr.responseText);

        for (let unit of Object.keys(dpu_data)) {
            dataType = unit.split('-')[0]
            dataSubType = unit.split('-')[1]

            for (let value of dpu_data[unit]) {
                loadData(dataType, dataSubType, value)
            }
        };

        console.log("Data load finished");

        //if (XIR_TGT.length <= 1) {
	if (true) {
            document.getElementById("timelineViewSelect").style.display = "none";
            document.getElementById("graphViewSelect").style.display = "none";
        } else if (XIR_LINKS.length == 0 || XIR_NODES.length == 0) {

        } else {

        }

        enableView("timeline");
    }
}

var URL = window.location.href;
taskId = parseInt(URL.split('id=')[1])
reload("/result/" + taskId);

function reload(url) {
    xhr.open('GET', url);
    xhr.overrideMimeType("text/html;charset=utf-8");
    xhr.send(null);
}

function loadData(type, subType, value) {
    switch (type) {
        case "TIME":
            if (subType == "start") {
                startTime = value
            }
            if (subType == "end") {
                endTime = value
            }
            break;
	case "TIMELINE":
            if (categories.indexOf(subType) == -1) {
                categories.push(subType)
            }

            title = value[0] + '-' + value[1]
            baseTime = value[2]
            endTime = value[3]
            duration = endTime - baseTime

            data.push({
                name: title,
                value: [
                    categories.indexOf(subType),
                    baseTime,
                    baseTime += duration,
                    duration
                ],
                itemStyle: {
                    normal: {
                        color: value[4]
                    }
                }
            });
            break;
        case "APM":
            axiTraffic.push(value);
            break;
        case "CG":
            cg_data.push(value)
            break
        case "FPS":
            FPS.push(value)
            break;
	case "INFO":
            title = "<strong>" + value['type'] + "</strong>";
            hwInfoData.push({ id: title })
            for (let item of Object.keys(value['info'])) {
                hwInfoData.push({ id: item, value: value['info'][item] })
            }
            break;
        case "xir":
            if (subType == "links") {
                XIR_LINKS.push(
                    value
                )
            }
            if (subType == "nodes") {
                XIR_NODES.push(
                    value
                )
            }
            if (subType == "tgt") {
                XIR_TGT.push(
                    value
                )
            }
            break;
	case "ENV":
	    break;
        default:
            console.log("data type error")
    }
}

function renderTimeline(params, api) {
    var categoryIndex = api.value(0);
    var start = api.coord([api.value(1), categoryIndex]);
    var end = api.coord([api.value(2), categoryIndex]);
    var height = api.size([0, 1])[1] * 0.3;

    var rectShape = echarts.graphic.clipRectByRect({
        x: start[0],
        y: start[1] - height / 2,
        width: end[0] - start[0],
        height: height
    }, {
        x: params.coordSys.x,
        y: params.coordSys.y,
        width: params.coordSys.width,
        height: params.coordSys.height
    });

    return rectShape && {
        type: 'rect',
        shape: rectShape,
        style: api.style()
    };
}

optionTimeline = {
    legend: [{
        show: false,
        data: ["DDRC_PORT_S1", "DDRC_PORT_S2", "DDRC_PORT_S3", "DDRC_PORT_S4", "DDRC_PORT_S5"],
        right: "10%",
        top: "68%",
        selectedMode: "single",
        tooltip: {
            show: true,
            formatter: function(para) {
                for (let a of APMMap) {
                    if (a['port'] == para.name) {
                        return a['description'];
                    }
                }
                return "error";
            }
        }
    }],
    tooltip: {
        //        formatter: function (params) {
        //            return params.marker + params.name + ': ' + Math.round(params.value[3] * 1000 * 1000) + ' us';
        //        }
    },
    title: {
        text: 'Xilinx Vitis AI Profile: Timeline',
        left: 'center',
    },
    dataZoom: [{
        type: 'slider',
        filterMode: 'empty',
        showDataShadow: false,
        top: 488,
        height: 5,
        borderColor: 'transparent',
        backgroundColor: '#e2e2e2',
        handleIcon: 'M10.7,11.9H9.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4h1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7v-1.2h6.6z M13.3,22H6.7v-1.2h6.6z M13.3,19.6H6.7v-1.2h6.6z', // jshint ignore:line
        handleSize: 10,
        handleStyle: {
            shadowBlur: 6,
            shadowOffsetX: 1,
            shadowOffsetY: 2,
            shadowColor: '#aaa'
        },
        //labelFormatter: '',
        xAxisIndex: [0, 1, 2]
    }, {
        type: 'inside',
        filterMode: 'empty',
        xAxisIndex: [0, 1, 2]
    }],
    grid: [{
            id: 'CUs',
            tooltip: {
                formatter: function(params) {
                    return params.marker + params.name + ': ' + Math.round(params.value[3] * 1000 * 1000) + ' us';
                }
            }
        }, {
            id: 'framerate',
            tooltip: {
                //trigger: 'axis',
                formatter: function(params) {
                    return (params.value[1]).toFixed(2) + "fps";
                },
                axisPointer: {
                    show: true,
                    snap: true,
                }
            },
        },
        {
            id: 'axiTraffic',
            tooltip: {
                formatter: function(params) {
                    return (params.value[1]).toFixed(2) + "MB/s";
                },
            },
        }
    ],
    xAxis: [{
        id: 'CUs',
        gridIndex: 0,
        scale: true,
        axisLabel: {
            //    formatter: function (val) {
            //        return Math.round(Math.max(0, val - startTime) * 1000) + ' ms';
            //    },
            show: false
        }
    }, {
        id: 'framerate',
        gridIndex: 1,
        axisLabel: {
            //formatter: function (val) {
            //    return Math.round(Math.max(0, val - startTime) * 1000) + ' ms';
            //},
            show: false
        },
        scale: true,
        show: true
    }, {
        id: 'axiTraffic',
        gridIndex: 2,
        axisLabel: {
            show: false
        },
        scale: true,
        show: true
    }],
    yAxis: [{
        id: "CUs",
        gridIndex: 0,
        data: categories
    }, {
        id: 'framerate',
        gridIndex: 1,
        show: true,
        name: "Throughput(FPS)",
        nameLocation: "middle",
        nameGap: 37,
        splitNumber: 4,
        type: 'value',
        min: 0,
        max: function(value) {
            return value.max * 1.2;
        },
        axisTick: { show: false },
        axisLine: { show: false },
        axisLabel: {
            formatter: function(value, index) {
                return value.toFixed(0);
            }
        }
    }, {
        id: 'axiTraffic',
        gridIndex: 2,
        show: true,
        name: "Total DDR Traffic",
        nameLocation: "middle",
        type: 'value',
        min: 0,
        //max: 18000,
        max: function(value) {
            return (value.max < 1000) ? 1100 : value.max * 1.1;
        },
        axisTick: { show: false },
        axisLine: { show: false },
        axisLabel: { show: false }
    }],
    series: [{
        type: 'custom',
        renderItem: renderTimeline,
        itemStyle: {
            normal: {
                opacity: 0.8
            }
        },
        encode: {
            x: [1, 2],
            y: 0
        },
        data: data,
        animation: false
    }, {
        type: 'line',
        xAxisIndex: 1,
        yAxisIndex: 1,
        barWidth: 3,
        data: FPS,
        symbolSize: function(value, params) {
            len = FPS.length;
            if (len > 200) {
                return 2;
            } else if (len > 100) {
                return 3;
            } else {
                return 4;
            }
        },
        animation: false,
        smooth: true
    }, {
        type: 'bar',
        name: "DDRC_PORT_S1",
        xAxisIndex: 2,
        yAxisIndex: 2,
        barWidth: 3,
        data: axiTraffic,
        encode: {
            x: 0,
            y: 1
        },
        tooltip: {
            formatter: function(params) {
                return (params.value[params.seriesIndex - 1]).toFixed(2) + "MB/s";
            }
        },
        animation: false
    }, {
        type: 'bar',
        name: "DDRC_PORT_S2",
        xAxisIndex: 2,
        yAxisIndex: 2,
        barWidth: 3,
        data: axiTraffic,
        encode: {
            x: 0,
            y: 2
        },
        tooltip: {
            formatter: function(params) {
                return (params.value[params.seriesIndex - 1]).toFixed(2) + "MB/s";
            }
        },
        animation: false
    }, {
        type: 'bar',
        name: "DDRC_PORT_S3",
        xAxisIndex: 2,
        yAxisIndex: 2,
        barWidth: 3,
        data: axiTraffic,
        encode: {
            x: 0,
            y: 3
        },
        tooltip: {
            formatter: function(params) {
                return (params.value[params.seriesIndex - 1]).toFixed(2) + "MB/s";
            }
        },
        animation: false
    }, {
        type: 'bar',
        name: "DDRC_PORT_S4",
        xAxisIndex: 2,
        yAxisIndex: 2,
        barWidth: 3,
        data: axiTraffic,
        encode: {
            x: 0,
            y: 4
        },
        tooltip: {
            formatter: function(params) {
                return (params.value[params.seriesIndex - 1]).toFixed(2) + "MB/s";
            }
        },
        animation: false
    }, {
        type: 'bar',
        name: "DDRC_PORT_S5",
        xAxisIndex: 2,
        yAxisIndex: 2,
        barWidth: 3,
        data: axiTraffic,
        encode: {
            x: 0,
            y: 5
        },
        tooltip: {
            formatter: function(params) {
                return (params.value[params.seriesIndex - 1]).toFixed(2) + "MB/s";
            }
        },
        animation: false
    }]
};

optionXir = {
    title: {
        text: 'Graph_0'
    },
    tooltip: {
        show: true,
        formatter: function(params) {
            return "Name: " + params.data.title + "<br />" +
                "Device: " + params.data.category + "<br />"
        }
    },
    animation: false,
    series: [{
        type: 'graph',
        layout: 'none',
        symbol: 'rect',
        symbolSize: [120, 30],
        roam: true,
        label: {
            show: true
        },
        edgeSymbol: ['circle', 'arrow'],
        edgeSymbolSize: [10, 10],
        edgeLabel: {
            fontSize: 15
        },
        categories: [{
            name: 'user',
            itemStyle: {
                color: 'green'
            }
        }, {
            name: 'dpu',
            itemStyle: {
                color: 'blue'
            }
        }, {
            name: 'cpu',
            itemStyle: {
                color: 'red'
            }
        }],
        nodes: XIR_NODES,
        links: XIR_LINKS
    }]
};

function showTreeGridTable(id) {
    cg = document.getElementById("cgTreeTable");
    xir = document.getElementById("xirTreeTable");

    if (id == "cg") {
        cg.style.display = "";
        xir.style.display = "none";
    } else if (id == "xir") {
        cg.style.display = "none";
        xir.style.display = "";
    }
}

$(window).on('resize', function() {
    var x = $(window).width();
    console.log("window width" + x);
    console.log("chart width" + vaiProfilerChart.getWidth());
    vaiProfilerChart.resize();
    console.log("new chart width" + vaiProfilerChart.getWidth());
}).resize();

function fancytreeSetFocus(treeid, key) {
    var tree = $.ui.fancytree.getTree(treeid);
    var node = tree.getNodeByKey(key);

    node.setActive(true);
    node.setExpanded(true);
    node.collapseSiblings();
    opts = {
        start: node._rowIdx
    };
    tree.setViewport(opts);
    $.ui.fancytree.getTree().redrawViewport(true);
}


function prepareChart(id) {
    vaiProfilerChart.on('click', function(params) {
        if (params.componentSubType == 'graph') {
            var key = params.data.title;
            fancytreeSetFocus("#xirTreeTable", key);

        } else {
            var taskPid = params.name;
            var startTime = params.value[1];
            var cgKey = ""

            //Walk thought cg_data find out the key
            for (let thread of cg_data) {
                if (thread.title.split('-')[1] == taskPid.split('-')[1]) {
                    for (let task of thread['children']) {
                        if (task['startTime'] <= startTime && task['endTime'] >= startTime) {
                            cgKey = task['key'];
                            console.log("Jump to task Key: " + task['key']);
                            fancytreeSetFocus("#cgTreeTable", cgKey);
                            //var tree = $.ui.fancytree.getTree();
                            //var node = tree.getNodeByKey(cgKey);
                            //node.setActive(true);
                            //node.setExpanded(true);
                            //node.collapseSiblings();
                            //opts = {
                            //    start: node._rowIdx
                            //};
                            //tree.setViewport(opts);
                            //$.ui.fancytree.getTree().redrawViewport(true);
                        }
                    }
                }
            }
            if (cgKey == "") {
                console.error("Cannot find the task in cg table");
            }
        }
    });

    //Set start point and end point
    for (let xaxis of optionTimeline.xAxis) {
        xaxis.min = startTime;
        xaxis.max = endTime
    }

    //Layout Adjuest
    areaTotalHeight = $("#mainChart").height();
    showAxiTraffic = (axiTraffic.length != 0);
    showFPS = (FPS.length != 0);

    /* Grids */
    FPSHight = showFPS ? 80 : 0;
    axiTrafficHight = showAxiTraffic ? 80 : 0;
    CUsHight = 200 + !showFPS * 80 + !showAxiTraffic * 80;

    CUsTop = 50;
    FPSTop = CUsHight + 80;
    axiTrafficTop = FPSTop + FPSHight + 25;

    // CUs
    optionTimeline['grid'][0]['top'] = CUsTop;
    optionTimeline['grid'][0]['height'] = CUsHight;
    // FPS
    optionTimeline['grid'][1]['top'] = FPSTop;
    optionTimeline['grid'][1]['height'] = FPSHight;
    // AXI Traffic
    optionTimeline['grid'][2]['top'] = axiTrafficTop;
    optionTimeline['grid'][2]['height'] = axiTrafficHight;

    /* legend */
    optionTimeline['legend'][0]['show'] = showAxiTraffic;

    /* yAxis */
    optionTimeline['yAxis'][1]['show'] = showFPS;
    optionTimeline['yAxis'][2]['show'] = showAxiTraffic;

    /* xAxis */
    timelineAxisLabel = {
        formatter: function(val) {
            return (val - startTime).toFixed(3) + ' s';
        }
    }
    optionTimeline['xAxis'][1]['show'] = showFPS;
    optionTimeline['xAxis'][2]['show'] = showAxiTraffic;

    if (optionTimeline['xAxis'][2]['show']) {
        optionTimeline['xAxis'][2]['axisLabel'] = timelineAxisLabel;
    } else if (optionTimeline['xAxis'][1]['show']) {
        optionTimeline['xAxis'][1]['axisLabel'] = timelineAxisLabel;
    } else {
        optionTimeline['xAxis'][0]['axisLabel'] = timelineAxisLabel;
    };

    vaiProfilerChart.hideLoading();

    console.log("eCharts init finished")
}

function prepareTGTable(id) {
    console.log("Tree-Grid-Table preparing: " + id)

    //if (id == 'timeline') {
    if (cg_data.length > 0) {
        $("#cgTreeTable").fancytree({
            extensions: ["grid"],
            autoScroll: true,
            autoScroll: true,
            table: {
                indentation: 16,
                nodeColumnIdx: 0
            },
            viewport: {
                enabled: true,
                count: 11,
            },
            tooltip: function(event, data) {
                //return data.node.data.startTime;
            },
            renderColumns: function(event, data) {
                var node = data.node,
                    $tdList = $(node.tr).find(">td");

                // (index #0 is rendered by fancytree by adding the checkbox)
                if (node.data.isHw) {
                    $tdList.eq(1).text("DPU");
                } else {
                    $tdList.eq(1).text(" ");
                }
                //$tdList.eq(1).text(node.getIndexHier());
                $tdList.eq(2).text(node.data.startTime.toFixed(6));
                $tdList.eq(3).text(node.data.endTime.toFixed(6));
                $tdList.eq(4).text(node.data.durnation);
            },
            updateViewport: function(event, data) {
                var tree = data.tree,
                    topNode = tree.visibleNodeList[tree.viewport.start];
                console.log(tree.viewport.start);
            },
            source: cg_data
        });
    }

    //if (id == "graph") {
    //if (XIR_TGT.length > 1) {
    if (false) {
        $("#xirTreeTable").fancytree({
            extensions: ["grid"],
            autoScroll: true,
            autoScroll: true,
            table: {
                indentation: 16,
                nodeColumnIdx: 0
            },
            viewport: {
                enabled: true,
                count: 11,
            },

            renderColumns: function(event, data) {
                var node = data.node,
                    $tdList = $(node.tr).find(">td");

                $tdList.eq(2).text("");

                if (node.data.timeMax == 0) {
                    elapsedMax = ""
                } else {
                    elapsedMax = node.data.timeMax.toFixed(2)
                }
                $tdList.eq(2).text(elapsedMax);

                if (node.data.timeMin == 0) {
                    elapsedMin = ""
                } else {
                    elapsedMin = node.data.timeMin.toFixed(2)
                }
                $tdList.eq(3).text(elapsedMin);

                if (node.data.time == 0) {
                    elapsed = ""
                } else {
                    elapsed = node.data.time.toFixed(2)
                }

                $tdList.eq(4).text(elapsed);
                // $tdList.eq(4).text(node.data[4]);
                // $tdList.eq(5).text(node.data[5]);
            },
            //updateViewport: function(event, data) {
            //    var tree = data.tree,
            //        topNode = tree.visibleNodeList[tree.viewport.start];
            //    console.log(tree.viewport.start);
            //},
            source: XIR_TGT
        });
    }
    console.log("Tree-Grid-Table prepare finished")
}

function prepareInfoTable(id) {
    layui.use('table', function() {
        var infoTable = layui.table;
        infoTable.render({
            elem: '#infoTable',
            height: 'full-100',
            width: 400,
            limit: 1000,
            data: hwInfoData,
            page: false,
            cols: [
                [
                    { field: 'id', title: 'Item', width: 250, align: 'left' },
                    { field: 'value', title: 'Value' }
                ]
            ],
            done: function(res, curr, count) {
                console.log("Info table init finished")
            }
        });
    });

    console.log("Info-Table prepare finished");
}

var views = {
    'timeline': {
        'profilerChartOption': optionTimeline,
        'treeGridTableName': "cg",
        'infoTableOption': 0
    },
    'graph': {
        'profilerChartOption': optionTimeline,
        'treeGridTableName': "xir",
        'infoTableOption': 0
    }
};

prepared = [];

function prepareView(viewName) {

    id = prepared.indexOf(viewName)
    if (id == -1) {
        console.log("Preparing view" + viewName)
        prepareTGTable(viewName);
        prepareInfoTable(viewName);
        prepareChart(viewName);
        prepared.push(viewName);
    }
}

function showChart(id) {
    if (id == 'timeline') {
        vaiProfilerChart.setOption(optionTimeline, notMerge = true);
    } else if (id == "graph") {
        //vaiProfilerChart.setOption(optionXir, notMerge = true);
    }
}

function showView(viewName) {
    console.log("Showing view" + viewName)
    tgt = views[viewName]['treeGridTableName'];
    showTreeGridTable(tgt);
    showChart(viewName)
}

function enableView(viewName) {
    console.log("Enabling view" + viewName)
    prepareView(viewName);
    showView(viewName);
}
