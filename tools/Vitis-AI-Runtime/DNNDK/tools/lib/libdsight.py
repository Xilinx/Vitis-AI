'''
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

# -*- coding:utf-8 -*-
from __future__ import division
import os, sys

if len(sys.argv) != 2:
    print 'Plz enter one filepath'
    sys.exit(0)

pwd = sys.argv[1]


if os.path.exists(pwd) == False:
    print '[DSight]Error - DPU trace file not exist!'
    sys.exit(0)
outputname = pwd.split('/')[-1]

outputname = outputname.split('.')
if outputname[1] != 'prof':
    print '[DSight]Error - invalid format DPU trace file specified!'
    sys.exit(0)
outputname = outputname[0]

#print 'prof data: ', pwd

color = []
color.append([0, 174, 187])
color.append([255,44,14]) 
color.append([123,177,55]) 
color.append([250, 202 ,46])
color.append([0, 56, 126]) 






# color.append([117,207,219]) 


color.append([216,0,102]) 
color.append([240,195,186]) 

color.append([219,178,209]) 
color.append([190,215,54]) 
color.append([227,65,50]) 
color.append([0,164,143]) 
color.append([100,83,148]) 
color.append([141,61,56]) 



color.append([7,61,81]) 
color.append([188,108,167]) 
color.append([58,116,150]) 
color.append([178,181,185]) 
color.append([69,97,51]) 
color.append([102,77,59]) 
color.append([202,160,102]) 
posit = []
posit.append([0,100])
posit.append([0,100])
posit.append([0,100])
posit.append([0,100])

dirpath = os.path.dirname(pwd)

profile_file1 = file(pwd)
x =  len(profile_file1.readlines())
if x < 1:
    print '[DSight]Error - invilad format DPU trace file specified!'
    sys.exit(0)
profile_file1.close()
if x > 5000:
    x = x - 5000
else:
    x = 0
# print x
profile_file = file(pwd)

core_count = 0
kernel_count = 0
core_list = set()
kernel_list = set()
kernel_list_unique = set()
# min_value = {}
min_value = 9223372036854775807
max_value = 0
core = []
x1 = 0
for line_ in profile_file:
    line_ = line_.strip()
    line_ = line_.split()
    x1 = x1 + 1
    if x1 < x:
        continue
    if  len(line_) != 5:
        print '[DSight]Error - invilad format DPU trace file specified!'
        sys.exit(0)
    if line_[2].isdigit() == False:
        print '[DSight]Error - invilad format DPU trace file specified!'
        sys.exit(0)
    if line_[3].isdigit() == False:
        print '[DSight]Error - invilad format DPU trace file specified!'
        sys.exit(0)
    core_list.add(line_[0])
    # kernel_list.add(line_[1])
    core.append(line_)




# kernel_count = len(kernel_list)
core_list = sorted(list(core_list))
if len(core_list) > 3:
    print "[DSight]Error - unsupported DPUCore number"
    sys.exit(0)
# kernel_list = sorted(list(kernel_list))

# for i in core_list:
#     min_value[i] = sys.maxint
for line_ in core:
    # if min_value[line_[0]] > int(line_[2]):
    #     min_value[line_[0]] = int(line_[2])
    if min_value > int(line_[2]):
        min_value = int(line_[2])
    # if max_value < int(line_[3]):
    #     max_value = int(line_[3])
# #print max_value
# max_value = int(max_value + (max_value - min_value['0'])/20)





show_info = []
for line_ in core:
    kernel_list.add("Core"+line_[0]+"-"+line_[1])
    kernel_list_unique.add(line_[1])
    tmp = [line_[0], "Core"+line_[0]+"-"+line_[1], int(line_[2]) - min_value, int(line_[3]) - min_value, line_[4]]
    ##print tmp
    show_info.append(tmp)
# print kernel_list
effe = {}
effe_real = {}
for ind, item in enumerate(core_list):
    effe[item] = 0
    effe_real[item] = 0

kernel_list = sorted(list(kernel_list))

color_map = {}
core_name = ""
if len(kernel_list_unique) > 20:
    print "[DSight]Error - unsupported DPUKernel number"
    sys.exit(1)

for ind, item in enumerate(kernel_list_unique):
    tmp =  color[ind]
    color_map[item] = tmp
for ind,item in enumerate(kernel_list):
    # tmp =  color[ind]
    # color_map[item] = tmp
    # print item.split('-')[1]
    core_name =core_name + "{name:'" + item +"',icon:'path://M16 6c-6.979 0-13.028 4.064-16 10 2.972 5.936 9.021 10 16 10s13.027-4.064 16-10c-2.972-5.936-9.021-10-16-10zM23.889 11.303c1.88 1.199 3.473 2.805 4.67 4.697-1.197 1.891-2.79 3.498-4.67 4.697-2.362 1.507-5.090 2.303-7.889 2.303s-5.527-0.796-7.889-2.303c-1.88-1.199-3.473-2.805-4.67-4.697 1.197-1.891 2.79-3.498 4.67-4.697 0.122-0.078 0.246-0.154 0.371-0.228-0.311 0.854-0.482 1.776-0.482 2.737 0 4.418 3.582 8 8 8s8-3.582 8-8c0-0.962-0.17-1.883-0.482-2.737 0.124 0.074 0.248 0.15 0.371 0.228v0zM16 13c0 1.657-1.343 3-3 3s-3-1.343-3-3 1.343-3 3-3 3 1.343 3 3z', textStyle:{color:'rgb(" + str(color_map[item.split('-')[1]][0]) + "," + str(color_map[item.split('-')[1]][1]) + "," +str(color_map[item.split('-')[1]][2])  + "',fontSize:'15'}}," 
#print 'core name',core_name 
#print color_map

# print color_map


for line_ in show_info:
    # if min_value[line_[0]] > int(line_[2]):
    #     min_value[line_[0]] = int(line_[2])
    # if min_value > int(line_[2]):
    #     min_value = int(line_[2])
    effe[line_[0]] = effe[line_[0]] + line_[3] - line_[2]
    effe_real[line_[0]] = effe_real[line_[0]] + (line_[3] - line_[2])*float(line_[4])
    if max_value < int(line_[3]):
        max_value = int(line_[3])
#print max_value


#print effe,"effe"

all = 0

posit_map = {}
for ind, item in enumerate(core_list):
    all = all + effe[item]
    effe_real[item] = effe_real[item] / effe[item]
    effe[item] = effe[item] / max_value

    posit_map[item] = posit[ind]
    
# print len(core_list)
#print  str(len(posit_map)*2)

per_str = ""
kernel_name = ""
aver = 0
max_value = int(max_value + (max_value)/20)
echart = ""
if os.path.exists("/usr/lib/echarts.js") == False:
    print '[DSight]Error - can not found echarts.js in /usr/lib/!'
    sys.exit(0)
echarts = file("/usr/lib/echarts.js")
for line in echarts:
    echart = echart + line

per_str = per_str + "       DPU          Utilization: "
for ind, item in enumerate(core_list):
    

    if round(effe_real[item]*100,1) >= 10:
        per_str=per_str + "Core" + item +": " + str(round(effe_real[item]*100,1))+"%  "
    else:
        per_str=per_str + "Core" + item +":   " + str(round(effe_real[item]*100,1))+"%  "

per_str = per_str + "\\n "

per_str = per_str + "       Schedule  Effeciency: "
for ind, item in enumerate(core_list):
    
    if round(effe[item]*100,1) >= 10:
        per_str=per_str + "Core" + item +": " + str(round(effe[item]*100,1))+"%  "
    else:
        per_str=per_str + "Core" + item +":   " + str(round(effe[item]*100,1))+"%  "

aver = all / (max_value * len(core_list))
#per_str = per_str + " All: " + str(round(aver*100, 1)) + "%" 
max_value = '\n        max: ' + str(max_value) + ', \n'

gridbox=""
xAxisbox=""
yAxisbox=""

xAxisIndex = ""
for ind, item in enumerate(core_list):
    xAxisIndex = xAxisIndex + str(ind) + ","

if(len(core_list) == 1):
    gridbox=gridbox + """
    {
        show: true,
        left: '2%',
        bottom: '""" + str(10) +"""%',
        right: '16%',
        top: '""" + str(35) +"""%',
        containLabel: true
    },            
    """
    xAxisbox=xAxisbox + """
    {
        type: 'value',
        name: 'Timeline(us)',
        gridIndex:0,

        nameTextStyle: {
            fontStyle: 'italic',
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        nameGap: 20,
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
        min: 0,"""+max_value+"""


        zlevel: 0
    },
    """
    yAxisbox = yAxisbox + """
    {
        type: 'value',
        min: 0,
        max: 100,
        zlevel: 0,
        name: 'Core0-Utilization(%)',
        axisTick: {show: false},
        axisLabel: {show: false},
        nameLocation: 'middle',
        splitNumber:1,
        gridIndex:0,
        nameTextStyle: {
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        axisLabel:{
            show:false
        },
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
    },
    """
    
elif(len(core_list) == 2):
    gridbox=gridbox + """
    {
        show: true,
        left: '2%',
        bottom: '""" + str(45) +"""%',
        right: '16%',
        top: '""" + str(20) +"""%',
        containLabel: true
    },            
    """
    gridbox=gridbox + """
    {
        show: true,
        left: '2%',
        bottom: '""" + str(10) +"""%',
        right: '16%',
        top: '""" + str(55) +"""%',
        containLabel: true
    },            
    """
    xAxisbox=xAxisbox + """
    {
        type: 'value',
        axisTick: {show: false},
        axisLabel: {show: false},
        axisLabel:{
            show:false
        },
        gridIndex:0,
        nameTextStyle: {
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        nameGap: 20,
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
        min: 0,"""+max_value+"""


        zlevel: 0
    },
    {
        type: 'value',
        gridIndex:1,

        name: 'Timeline(us)',

        nameTextStyle: {
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        nameGap: 20,
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
        min: 0,"""+max_value+"""


        zlevel: 0
    },
    """
    yAxisbox = yAxisbox + """
    {
        type: 'value',
        min: 0,
        max: 100,
        zlevel: 0,
        splitNumber:1,
        name: 'Core0-Utilization(%)',
        nameLocation: 'middle',
        gridIndex:0,
        axisTick: {show: false},
        axisLabel: {show: false},
        axisLabel:{
            show:false
        },
        nameTextStyle: {
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
    },
    {
        type: 'value',
        min: 0,
        max: 100,
        zlevel: 0,
        name: 'Core1-Utilization(%)',
        axisTick: {show: false},
        axisLabel: {show: false},
        nameLocation: 'middle',
        splitNumber:1,
        gridIndex:1,
        nameTextStyle: {
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        axisLabel:{
            show:false
        },
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
    },
    """

elif(len(core_list) == 3):
    gridbox=gridbox + """
    {
        show: true,
        left: '2%',
        bottom: '""" + str(57) +"""%',
        right: '16%',
        top: '""" + str(20) +"""%',
        containLabel: true
    },            
    """
    gridbox=gridbox + """
    {
        show: true,
        left: '2%',
        bottom: '""" + str(34) +"""%',
        right: '16%',
        top: '""" + str(43) +"""%',
        containLabel: true
    },            
    """
    gridbox=gridbox + """
    {
        show: true,
        left: '2%',
        bottom: '""" + str(10) +"""%',
        right: '16%',
        top: '""" + str(66) +"""%',
        containLabel: true
    },            
    """
    xAxisbox=xAxisbox + """
    {
        type: 'value',
        axisTick: {show: false},
        axisLabel: {show: false},
        axisLabel:{
            show:false
        },
        gridIndex:0,
        nameTextStyle: {
            fontStyle: 'italic',
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        nameGap: 20,
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
        min: 0,"""+max_value+"""


        zlevel: 0
    },
    {
        type: 'value',
        gridIndex:1,
        axisLabel:{
            show:false
        },


        nameTextStyle: {
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        nameGap: 20,
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
        min: 0,"""+max_value+"""


        zlevel: 0
    },
    {
        type: 'value',
        gridIndex:2,

        name: 'Timeline(us)',

        nameTextStyle: {
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        nameGap: 20,
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
        min: 0,"""+max_value+"""


        zlevel: 0
    },
    """
    yAxisbox = yAxisbox + """
    {
        type: 'value',
        min: 0,
        max: 100,
        zlevel: 0,
        splitNumber:1,
        name: 'Core0-Utilization(%)',
        nameLocation: 'middle',
        gridIndex:0,
        axisTick: {show: false},
        axisLabel: {show: false},
        axisLabel:{
            show:false
        },
        nameTextStyle: {
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
    },
    {
        type: 'value',
        min: 0,
        max: 100,
        zlevel: 0,
        name: 'Core1-Utilization(%)',
        axisTick: {show: false},
        axisLabel: {show: false},
        nameLocation: 'middle',
        splitNumber:1,
        gridIndex:1,
        nameTextStyle: {
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        axisLabel:{
            show:false
        },
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
    },
    {
        type: 'value',
        min: 0,
        max: 100,
        zlevel: 0,
        name: 'Core2-Utilization(%)',
        axisTick: {show: false},
        axisLabel: {show: false},
        nameLocation: 'middle',
        splitNumber:1,
        gridIndex:2,
        nameTextStyle: {
            fontWeight: 'bold',
            fontFamily: 'calibri',
            fontSize: 16,
        },
        axisLabel:{
            show:false
        },
        splitLine:{
            lineStyle:{
                type:'dashed'
            }
        },
    },
    """

else:
    print "Failed!"
    sys.exit(0)



jshead ="""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DSight chart</title>
    <script>""" + echart + """</script>
</head>
<body>
	
    <div id="main" style="width: 100%;height:700px;overflow: hidden;margin:0"></div>
    <script type="text/javascript">
        var myChart = echarts.init(document.getElementById('main'));
option = {
    animation: false,
        color:['rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)','rgb(89, 87, 87)',],
    legend:{
      type: 'scroll',
      textStyle:{
          fontFamily: 'calibri',
      },
      data:["""+core_name+"""]  ,
      bottom:'16%',
      orient:'vertical',
      left:'84.5%'
    },
    axisPointer: {
        show:true,
        z:100,
        lineStyle: {
            width:2,
            color:'rgb(240, 40, 40)',
        },
        link: [{xAxisIndex: 'all'}],
    },
    toolbox: {
        feature: {
            dataZoom: {
                yAxisIndex: 'none',
                iconStyle: {
                    normal: {
                        color:'rgba(0, 174, 187, 0.5)',
                    },
                },
            },
            restore: {
                show: true
            },
            saveAsImage: {
                show: true
            }
        }
    },
    tooltip:{
        trigger:'item',

    },
    toolbox: {
        feature: {

            restore: {
                show: true
            },
            saveAsImage: {
                show: true
            }
        }
    },
    title: {
        text:'    Xilinx DSight',
        subtext: ' """+per_str +"""',
        textStyle: {
            fontStyle: 'normal',
            fontFamily: 'calibri',
            fontSize: '40',
            color: 'rgb(0, 174, 187)'
        },
        subtextStyle: {
            fontStyle: 'normal',
            fontFamily: 'calibri',
            fontSize: '20',
            color: 'rgb(62, 28, 28)'
        }

    },
    grid: [""" + gridbox + """
    ],
    yAxis: [""" + yAxisbox + """
    ],
    xAxis: [ """ + xAxisbox +  """
        
    ],

    dataZoom: [{
        type: 'slider',
        filterMode:'empty',
        realtime:false,
        xAxisIndex: [""" + xAxisIndex + """]

    }, {
        type: 'inside',
        filterMode:'empty',
        realtime:false,
        xAxisIndex: [""" + xAxisIndex + """]
    }],
    series: ["""




mark = []
me = []

core_list_ind = {}
# core_list.sort(reverse = True)
for ind,item in enumerate(core_list):
    core_list_ind[item] = ind
# print core_list_ind
for ind,item in enumerate(kernel_list):
    core_ind = item.split('-')[0][-1]
    core_ind = core_list_ind[core_ind]
    me_ = """
                ],
            },
            zlevel:  """ + str(ind + 1)  + """
            },
    """
    me.append(me_)
    mark_ = """
			{
            name: '""" + item + """',
            type: 'line',
            data: [],
            xAxisIndex: """+ str(core_ind) +""",
            yAxisIndex: """+ str(core_ind) +""",  
            markArea: {
                itemStyle: {
                    normal: {
                        //color:'#4eea94'
                        //color: '#00ff7f'
                        //color:'rgba(32,205,32,1)'
                    }
                },
                data: [
                    """    
    mark.append(mark_)
    #print mark_

hend = """
    ]
};


		myChart.setOption(option);
		window.onresize = myChart.resize;
    </script>
</body>
</html>
"""

markline = """
{
            name:'line',
            type: 'line',

            markLine: {
                symbol: 'diamond',
                symbolSize: 2,
                lineStyle:{
                    normal:{
                        width:1,
                        type:'solid',
                        color:'rgba(25,25,112,1)',

                    },
                    emphasis:{
						width:2,
                        color:'rgba(25,25,112,1)',

                    },
                },

                data: [
                    [{coord:[0,1.8]},{coord:[16000000,1.8]}],
                                    ],
            },
            zlevel:0
        },

"""

if dirpath == '':
    dirpath = './'
else:
    dirpath = dirpath + '/'
htmlout = file(dirpath  + outputname +".html", "w")
htmlout.writelines(jshead)
# htmlout.writelines(max_value)
# htmlout.writelines(jsmid)


for ind,item in enumerate(kernel_list):
    htmlout.writelines(mark[ind])
    for info in show_info:
        if info[1] == item:
            tmp = (posit_map[info[0]][1] - posit_map[info[0]][0])*(1 - float(info[4]))
            htmlout.writelines("[{coord:[" + str(info[2]) + "," + str(posit_map[info[0]][0]) + "],itemStyle: {normal: { color:'rgba(" +str(color_map[item.split('-')[1]][0]) +"," + str(color_map[item.split('-')[1]][1]) + "," +str(color_map[item.split('-')[1]][2]) + ",1" + ")'}},},{coord:[" +str(info[3]) + "," + str(posit_map[info[0]][1] - tmp) + "]}],\n")

    htmlout.writelines(me[ind])
# htmlout.writelines(markline) 
htmlout.writelines(hend) 
htmlout.close()
# print 'COMPLETE!!!'

print "Generate DPU's visualization profiling chart: " + os.path.abspath(dirpath  + outputname +".html")
