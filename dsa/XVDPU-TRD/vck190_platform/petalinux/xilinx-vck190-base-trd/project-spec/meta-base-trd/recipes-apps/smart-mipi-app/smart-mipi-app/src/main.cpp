/*
 * Copyright 2021 Xilinx Inc.
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

#include <errno.h>
#include <fcntl.h>
#include <gst/gst.h>
#include <iostream>
#include <linux/v4l2-subdev.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

#define assert2(cond, ...)                                                     \
    do {                                                                       \
        if (!(cond)) {                                                         \
            int errsv = errno;                                                 \
            fprintf(stderr, "ERROR(%s:%d) : ", __FILE__, __LINE__);            \
            errno = errsv;                                                     \
            fprintf(stderr, __VA_ARGS__);                                      \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))

struct sensor_reg {
    uint32_t id;
    int32_t val;
};

static const struct sensor_reg quad_mipi_ctrls[] = {
    {0x0098090e, 140}, // control 0x0098090e `AR0231 Red Balance' min 0 max 2047
                       // step 1 default 128 current 140.
    {0x0098090f, 500}, // control 0x0098090e `AR0231 Blue Balance' min 0 max
                       // 2047 step 1 default 128 current 140.
    {0x00980911, 1000}, // control 0x00980911 `AR0231 Exposure' min 16 max 1339
                        // step 1 default 821 current 1000.
    {0x00980913, 800},  // control 0x00980913 `AR0231 Digital Gain' min 0 max
                        // 2047 step 1 default 512 current 800.
    {0x00980924, 180},  // control 0x00980924 `AR0231 Green Balance' min 0 max
                        // 2047 step 1 default 145 current 180.`
};

// quad-mipi sensor v4l2 sub-device
static const char *quad_subdev[] = {
    "/dev/v4l-subdev14",
    "/dev/v4l-subdev13",
    "/dev/v4l-subdev12",
    "/dev/v4l-subdev11",
};

static const struct sensor_reg single_mipi_ctrls[] = {
    {0x00980911, 10000}, // control 0x00980911 `Exposure' min 14 max 16666 step 1 default 16666 current 9997
};

// single-mipi sensor v4l2 sub-device
static const char *single_subdev[] = {
    "/dev/v4l-subdev0",
};

struct PipelineInfo {
    gboolean single;
    gboolean verbose;
    gboolean performance;
    gboolean passthrough;
    gint channel_num;
    gint width;
    gint height;
    gint media_device;
    gchar *task;
    std::vector<gchar *> tasks;
    gchar *xclbin_location;
    gchar *config_dir;
};

PipelineInfo info = {false,
                     false,
                     false,
                     false,
                     4,
                     3840,
                     2160,
                     1,
                     (gchar *)"yolov3",
                     {(gchar *)"refinedet", (gchar *)"facedetect",
                      (gchar *)"ssd", (gchar *)"yolov3"},
                     (gchar *)"/media/sd-mmcblk0p1/binary_container_1.xclbin",
                     (gchar *)"/usr/share/ivas/smart-mipi-app"};

enum Channel { UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT, FULLSCREEN };

struct ChannelInfo {
    int plane_id;
    std::string input_format;
    std::vector<int> coordinate;
    std::string task;
};

std::vector<ChannelInfo> channels_info = {
    {34, "BGR", {0, 0, 1, 1}, info.tasks[0]},
    {35, "BGR", {1, 0, 1, 1}, info.tasks[1]},
    {36, "BGR", {0, 1, 1, 1}, info.tasks[2]},
    {37, "BGR", {1, 1, 1, 1}, info.tasks[3]},
    {34, "BGR", {0, 0, 1, 1}, info.task}};

std::string pipeline_string();
int get_config_info(int argc, char *argv[]);
int performance_test();

/* set subdevice control */
int v4l2_set_ctrl(char *subdev_name, int id, int value) {
    int fd, ret;
    struct v4l2_queryctrl query;
    struct v4l2_control ctrl;

    fd = open(subdev_name, O_RDWR);
    assert2(fd >= 0, "failed to open %s: %s\n", subdev_name, strerror(errno));

    memset(&query, 0, sizeof(query));
    query.id = id;
    ret = ioctl(fd, VIDIOC_QUERYCTRL, &query);
    assert2(ret >= 0, "VIDIOC_QUERYCTRL failed: %s\n", strerror(errno));

    if (query.flags & V4L2_CTRL_FLAG_DISABLED)
        printf("V4L2_CID_%d is disabled\n", id);
    else {
        memset(&ctrl, 0, sizeof(ctrl));
        ctrl.id = query.id;
        ctrl.value = value;
        ret = ioctl(fd, VIDIOC_S_CTRL, &ctrl);
        assert2(ret >= 0, "VIDIOC_S_CTRL failed: %s\n", strerror(errno));
    }

    close(fd);
    return 0;
}

int main(int argc, char *argv[]) {
    int m, n, ret;

    get_config_info(argc, argv);

    if (info.performance)
        return performance_test();

    /* fine-tune quad-mipi sensor for better quality */
    if (!info.single) {
        for (int m = 0; m < ARRAY_SIZE(quad_subdev); ++m) {
            for (n = 0; n < ARRAY_SIZE(quad_mipi_ctrls); ++n) {
                ret = v4l2_set_ctrl(quad_subdev[m], quad_mipi_ctrls[n].id,
                                    quad_mipi_ctrls[n].val);
                if (ret < 0)
                    continue;
            }
        }
    } else {
        for (n = 0; n < ARRAY_SIZE(single_mipi_ctrls); ++n) {
                    ret = v4l2_set_ctrl(single_subdev[0], single_mipi_ctrls[n].id,
                                        single_mipi_ctrls[n].val);
                    if (ret < 0)
                        continue;
                }
    }

    /* Initialize GStreamer */
    gst_init(&argc, &argv);

    /* Build the pipeline */
    auto pipeline = gst_parse_launch(pipeline_string().c_str(), NULL);

    /* Start playing */
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Wait until error or EOS */
    auto bus = gst_element_get_bus(pipeline);
    auto msg = gst_bus_timed_pop_filtered(
        bus, GST_CLOCK_TIME_NONE,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    /* Free resources */
    if (msg != NULL)
        gst_message_unref(msg);
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}

int get_config_info(int argc, char *argv[]) {
    gboolean silent = FALSE;
    gchar *savefile = NULL;
    GOptionContext *ctx;
    GError *err = NULL;
    GOptionEntry entries[] = {
        {"verbose", 'v', 0, G_OPTION_ARG_NONE, &info.verbose,
         "print gstreamer pipeline", NULL},
        {"single", 's', 0, G_OPTION_ARG_NONE, &info.single,
         "only process one channel video and display fullscreen", NULL},
        {"width", 'W', 0, G_OPTION_ARG_INT, &info.width,
         "resolution width of the input: [1920 | 3840], default: 3840",
         "WIDTH"},
        {"height", 'H', 0, G_OPTION_ARG_INT, &info.height,
         "resolution height of the input: [1080 | 2160], default: 2160",
         "HEIGHT"},
        {"task", 't', 0, G_OPTION_ARG_STRING, &info.task,
         "select AI task to be run: "
         "[yolov3 | facedetect | refinedet | ssd(adas)], default: "
         "yolov3, work only when single is true",
         "TASK"},
        {"t1", 0, 0, G_OPTION_ARG_STRING, &info.tasks[0],
         "select AI task to be run for channel 1, default: refinedet", "TASK"},
        {"t2", 0, 0, G_OPTION_ARG_STRING, &info.tasks[1],
         "select AI task to be run for channel 2, default: facedetect", "TASK"},
        {"t3", 0, 0, G_OPTION_ARG_STRING, &info.tasks[2],
         "select AI task to be run for channel 3, default: ssd(adas)", "TASK"},
        {"t4", 0, 0, G_OPTION_ARG_STRING, &info.tasks[3],
         "select AI task to be run for channel 4, default: yolov3", "TASK"},
        {"media-device", 'm', 0, G_OPTION_ARG_INT, &info.media_device,
         "num of media-device, default: 1 ", "NUM"},
        {"channel-num", 'n', 0, G_OPTION_ARG_INT, &info.channel_num,
         "channel numbers of video: [1 | 2 | 3 | 4], work only when single is "
         "false",
         "NUM"},
        {"xclbin-location", 'x', 0, G_OPTION_ARG_STRING, &info.xclbin_location,
         "set path of xclbin", "XCLBIN-LOCATION"},
        {"config-dir", 'c', 0, G_OPTION_ARG_STRING, &info.config_dir,
         "set config path of gstreamer plugin", "CONFIG-DIR"},
        {"performace", 'p', 0, G_OPTION_ARG_NONE, &info.performance,
         "print performance", NULL},
        {"passthrough", 'P', 0, G_OPTION_ARG_NONE, &info.passthrough,
         "check video passthrough, no ai task is running", NULL},
        {NULL}};

    ctx = g_option_context_new(
        "- Application for detction on VCK190 board of Xilinx. \n\n\
Examples for 4 mipi camera:\n\
  smart-mipi-app -m 2\n\
            # Run 4 channel mipi camera with 3840x2160 resolution monitor at /dev/media2.\n\
  smart-mipi-app -W 1920 -H 1080\n\
            # Change to 1920x1080 resolution monitor.\n\
  smart-mipi-app --t1=yolov3 --t2=refinedet --t3=facedetect --t4=ssd\n\
            # Change ai task for each channel.\n\
    \n\
Examples for single mipi camera:\n\
  smart-mipi-app -s -m 2\n\
            # Run single channel mipi camera with 3840x2160 resolution monitor at /dev/media2.\n\
  smart-mipi-app -s -W 1920 -H 1080\n\
            # Change to 1920x1080 resolution monitor.\n\
  smart-mipi-app -s -t ssd\n\
            # Change ai task from yolov3 to ssd.");

    g_option_context_add_main_entries(ctx, entries, NULL);
    g_option_context_add_group(ctx, gst_init_get_option_group());
    if (!g_option_context_parse(ctx, &argc, &argv, &err)) {
        g_print("Failed to initialize: %s\n", err->message);
        g_clear_error(&err);
        g_option_context_free(ctx);
        return 1;
    }
    g_option_context_free(ctx);

    // reset ai tasks
    for (int i = UP_LEFT; i <= DOWN_RIGHT; i++)
        channels_info[i].task = std::string(info.tasks[i]);

    channels_info[FULLSCREEN].task = std::string(info.task);

    return 0;
}

std::string get_channel_plane_id(Channel channel) {
    return std::to_string(channels_info[channel].plane_id);
}

std::string get_channel_input_format(Channel channel) {
    return channels_info[channel].input_format;
}

std::string get_channel_input_task(Channel channel) {
    return channels_info[channel].task;
}

std::string get_channel_coordinate(Channel channel) {
    std::string result;
    auto w = info.width / 2;
    auto h = info.height / 2;
    if (channel == FULLSCREEN) {
        w = info.width;
        h = info.height;
    }
    result = "\"<" + std::to_string(channels_info[channel].coordinate[0] * w) +
             "," + std::to_string(channels_info[channel].coordinate[1] * h) +
             "," + std::to_string(channels_info[channel].coordinate[2] * w) +
             "," + std::to_string(channels_info[channel].coordinate[3] * h) +
             ">\"";
    return result;
}

std::string get_task_pp_param(std::string model) {
    std::string result;
    if (model != "facedetect" && model != "yolov3" && model != "refinedet" &&
        model != "ssd") {
        g_printerr("ERROR: Not support task %s, only suport refinedet, "
                   "facedetect, ssd "
                   "and yolov3.\n",
                   model.c_str());
        abort();
    }
    if (model == "refinedet" || model == "ssd") {
        result = "alpha-r=\"104\" alpha-g=\"117\" alpha-b=\"123\" beta-r=\"1\" "
                 "beta-g=\"1\" beta-b=\"1\"";
    } else if (model == "facedetect") {
        result = "alpha-r=\"128\" alpha-g=\"128\" alpha-b=\"128\" beta-r=\"1\" "
                 "beta-g=\"1\" beta-b=\"1\"";
    } else {
        result = "alpha-r=\"0\" alpha-g=\"0\" alpha-b=\"0\" beta-r=\"0.25\" "
                 "beta-g=\"0.25\" beta-b=\"0.25\"";
    }

    return result;
}

std::string get_channel_input_string(Channel channel) {
    auto w = info.single ? info.width : info.width / 2;
    auto h = info.single ? info.height : info.height / 2;
    std::string result = " channel. ! video/x-raw, width=" + std::to_string(w) +
                         ", height=" + std::to_string(h) +
                         ", format=" + get_channel_input_format(channel) +
                         ", "
                         "framerate=60/1 ";
    return result;
}

std::string get_channel_inference_string(Channel channel) {
    std::string s = std::to_string(channel);
    std::string result =
        " ! tee name=t" + s +
        " ! ivas_xabrscaler xclbin-location=" + info.xclbin_location +
        " kernel-name=\"v_multi_scaler:v_multi_scaler_1\" " +
        get_task_pp_param(get_channel_input_task(channel)) +
        " ! queue ! ivas_xfilter kernels-config=" + info.config_dir + "/" +
        get_channel_input_task(channel) + "/aiinference.json \
        ! ima" +
        s + ".sink_master ivas_xmetaaffixer name=ima" + s + " ima" + s +
        ".src_master ! fakesink t" + s + ". ! queue ! ima" + s +
        ".sink_slave_0 ima" + s + ".src_slave_0 \
        ! queue ! ivas_xfilter kernels-config=" +
        info.config_dir + "/" + get_channel_input_task(channel) +
        "/drawresult.json ";
    return result;
}

std::string get_channel_display_string(Channel channel) {
    std::string result = " ! perf !  kmssink driver-name=xlnx plane-id=" +
                         get_channel_plane_id(channel) + " render-rectangle=" +
                         get_channel_coordinate(channel) + " sync=false ";
    return result;
}

std::string channel_string(Channel channel) {
    std::string result;

    if ((channel < info.channel_num || channel == FULLSCREEN) &&
        !info.passthrough)
        result = get_channel_input_string(channel) +
                 get_channel_inference_string(channel) +
                 get_channel_display_string(channel);
    else
        result = get_channel_input_string(channel) +
                 get_channel_display_string(channel);

    return result;
}

std::string single_channel_pipeline_string() {
    std::string result = "mediasrcbin media-device=/dev/media" +
                         std::to_string(info.media_device) + " name=channel " +
                         channel_string(static_cast<Channel>(FULLSCREEN));
    return result;
}

std::string multi_channel_pipeline_string() {
    std::string result = "mediasrcbin media-device=/dev/media" +
                         std::to_string(info.media_device) + " name=channel ";

    for (int i = UP_LEFT; i <= DOWN_RIGHT; i++)
        result += channel_string(static_cast<Channel>(i));

    return result;
}

std::string pipeline_string() {
    std::string result;
    if (info.single)
        result = single_channel_pipeline_string();
    else
        result = multi_channel_pipeline_string();

    if (info.verbose)
        std::cout << "gst-launch-1.0 " + result << "\n";

    return result;
}

int performance_test() {
    std::cout << "gst-launch-1.0 " + pipeline_string() << "\n";
    return system((std::string("gst-launch-1.0 ") + pipeline_string()).c_str());
}
