.. meta::
   :keywords: Vision, Library, Vitis Vision Library, Iterative Pyramidal, Corner Tracking, cornerUpdate, cornersImgToList, cv, mat
   :description: Vitis Vision library application programming interface reference.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _libapireference_aie:

xfcvDataMovers
##############

xfcvDataMovers class object takes input some simple parameters from users and provides a simple data transaction API where user does not have to bother about the complexity. Moreover it provides a template parameter using which application can switch from PL based data movement to  GMIO based (and vice versa) seamlessly. For more details please refer :ref:`xfcvDataMovers <xfcvdatamovers_aie>`.

Class Definition
================

.. code:: c

   template <DataMoverKind KIND,
             typename DATA_TYPE,
             int TILE_HEIGHT_MAX,
             int TILE_WIDTH_MAX,
             int AIE_VECTORIZATION_FACTOR,
             int CORES = 1,
             int PL_AXI_BITWIDTH = 32,
             bool USE_GMIO = false>
   class xfcvDataMovers {
       //Tiler  constructor
       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
       xfcvDataMovers(uint16_t overlapH, uint16_t overlapV);

       //Stitcher constructor
       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
       xfcvDataMovers();

       //Meta data computation
       void compute_metadata(const cv::Size& img_size);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
       std::array<uint16_t, 2> host2aie_nb(cv::Mat& img, xrtBufferHandle imgHndl = nullptr);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
       std::array<uint16_t, 2> host2aie_nb(xrtBufferHandle imgHndl, const cv::Size& size);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
       void aie2host_nb(cv::Mat& img, std::array<uint16_t, 2> tiles, xrtBufferHandle imgHndl = nullptr);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
       void aie2host_nb(xrtBufferHandle imgHndl, const cv::Size& size, std::array<uint16_t, 2> tiles);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
       void wait();

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
       void wait();;
   };

   //GMIO class specialization
   template <DataMoverKind KIND,
             typename DATA_TYPE,
             int TILE_HEIGHT_MAX,
             int TILE_WIDTH_MAX,
             int AIE_VECTORIZATION_FACTOR,
             int CORES>
   class xfcvDataMovers<KIND, DATA_TYPE, TILE_HEIGHT_MAX, TILE_WIDTH_MAX, AIE_VECTORIZATION_FACTOR, CORES, 0, true> {
       //Tiler  constructor
       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
       xfcvDataMovers(uint16_t overlapH, uint16_t overlapV);

       //Stitcher constructor
       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
       xfcvDataMovers();

       //Compute meta data
       void compute_metadata(const cv::Size& img_size);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
       std::array<uint16_t, 2> host2aie_nb(DATA_TYPE* img_data, const cv::Size& img_size, std::array<std::string, CORES> portNames);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
       std::array<uint16_t, 2> host2aie_nb(cv::Mat& img, std::array<std::string, CORES> portNames);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
       void aie2host_nb(DATA_TYPE* img_data, const cv::Size& img_size, std::array<uint16_t, 2> tiles, std::array<std::string, CORES> portNames);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
       void aie2host_nb(cv::Mat& img, std::array<uint16_t, 2> tiles, std::array<std::string, CORES> portNames);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
       std::array<uint16_t, 2> host2aie(cv::Mat& img, std::array<std::string, CORES> portNames);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
       std::array<uint16_t, 2> host2aie(DATA_TYPE* img_data, const cv::Size& img_size, std::array<std::string, CORES> portNames);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
       void aie2host(cv::Mat& img, std::array<uint16_t, 2> tiles, std::array<std::string, CORES> portNames);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
       void aie2host(DATA_TYPE* img_data, const cv::Size& img_size, std::array<uint16_t, 2> tiles, std::array<std::string, CORES> portNames);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == TILER)>::type* = nullptr>
       void wait(std::array<std::string, CORES> portNames);

       template <DataMoverKind _t = KIND, typename std::enable_if<(_t == STITCHER)>::type* = nullptr>
       void wait(std::array<std::string, CORES> portNames);
   };

.. table:: Table xF::xfcvDataMovers Member Function Descriptions

   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | Member                                                   | Description                                                                                 |
   | Functions                                                |                                                                                             |
   +==========================================================+=============================================================================================+
   | xfcvDataMovers(uint16_t overlapH, uint16_t overlapV)     | Tiler constructor using horizontal and vertical overlap sizes                               |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | xfcvDataMovers()                                         | Stitcher constructor                                                                        |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | host2aie_nb(cv::Mat& img,                                | Host to AIE non blocking transaction using input image.                                     |
   | xrtBufferHandle imgHndl = nullptr)                       |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | host2aie_nb(xrtBufferHandle imgHndl,                     | Host to AIE non blocking transaction using XRT allocated buffer handle and image size       |
   | const cv::Size& size)                                    |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | aie2host_nb(cv::Mat& img,                                | AIE to Host non blocking transaction using input image and {tile rows, tile cols} array     |
   | std::array<uint16_t, 2> tiles,                           |                                                                                             |
   | xrtBufferHandle imgHndl = nullptr)                       |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | aie2host_nb(xrtBufferHandle imgHndl,                     | AIE to Host non blocking transaction using XRT allocated buffer handle and image size       |
   | const cv::Size& size,                                    |                                                                                             |
   | std::array<uint16_t, 2> tiles)                           |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | wait()                                                   | Wait for transaction to complete                                                            |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+

.. note::
   If XRT mapped buffer handle is associated with image it can also be passed to imgHndl argument avoid copy

.. note::
   Parameter *tiles* can be obtained from tiler data transfer API host2aie_nb

.. table:: Table xF::xfcvDataMovers Member Function Descriptions (GMIO specialization)

   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | Member                                                   | Description                                                                                 |
   | Functions                                                |                                                                                             |
   +==========================================================+=============================================================================================+
   | xfcvDataMovers(uint16_t overlapH, uint16_t overlapV)     | Tiler constructor using horizontal and vertical overlap sizes                               |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | xfcvDataMovers()                                         | Stitcher constructor                                                                        |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | host2aie_nb(cv::Mat& img,                                | Host to AIE non blocking transaction using input image.                                     |
   | std::array<std::string, CORES> portNames)                |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | host2aie_nb(DATA_TYPE* img_data,                         | Host to AIE non blocking transaction using image data pointer and image size                |
   | const cv::Size& size,                                    |                                                                                             |
   | std::array<std::string, CORES> portNames)                |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | aie2host_nb(cv::Mat& img,                                | AIE to Host non blocking transaction using input image.                                     |
   | std::array<std::string, CORES> portNames)                |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | aie2host_nb(DATA_TYPE* img_data,                         | AIE to Host non blocking transaction using image data pointer and image size                |
   | const cv::Size& size,                                    |                                                                                             |
   | std::array<std::string, CORES> portNames)                |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | host2aie(cv::Mat& img,                                   | Host to AIE blocking transaction using input image.                                         |
   | std::array<std::string, CORES> portNames)                |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | host2aie(DATA_TYPE* img_data,                            | Host to AIE blocking transaction using image data pointer and image size                    |
   | const cv::Size& size,                                    |                                                                                             |
   | std::array<std::string, CORES> portNames)                |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | aie2host(cv::Mat& img,                                   | AIE to Host blocking transaction using input image.                                         |
   | std::array<std::string, CORES> portNames)                |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | aie2host(DATA_TYPE* img_data,                            | AIE to Host blocking transaction using image data pointer and image size                    |
   | const cv::Size& size,                                    |                                                                                             |
   | std::array<std::string, CORES> portNames)                |                                                                                             |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | wait()                                                   | Wait for transaction to complete                                                            |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+

.. note::
   Argument *portNames* correspond GMIO port declared as part of :ref:`platform specification <gmio_aie>`

.. _aie_library_functions:

Vitis Vision AIE Library Functions API list with performance estimates
#######################################################################

Frames per second (FPS) measured from host-code and includes data-transfer latencies and AIEngine™ kernel latencies. Measurements done on VCK190 evaluation boards and use only *one* AIE core.

.. table:: Table AIE Library Functions API list with performance estimates


   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | Function(xf::cv::aie)           | Performance (FPS) with PL Data-movers(Full HD) | Performance (FPS) with PL Data-movers (4K) | Performance (FPS) with GMIO Data-movers(Full HD) | Performance (FPS) with GMIO Data-movers (4K)  |
   +=================================+================================================+============================================+==================================================+===============================================+
   | absolutedifference              |                      284                       |                     80                     |                  128                             |                        50                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | accumulate                      |                      280                       |                     79                     |                  137                             |                        48                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | accumulateweighted              |                      281                       |                     75                     |                  111                             |                        40                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | addweighted                     |                      358                       |                    100                     |                  129                             |                        38                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | convertScaleAbs                 |                      406                       |                    106                     |                  214                             |                        57                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | erode                           |                      250                       |                     65                     |                  131                             |                        41                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | filter2D                        |                      250                       |                     67                     |                  157                             |                        47                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | gainControl                     |                      405                       |                    103                     |                  192                             |                        60                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | gaussian                        |                      249                       |                     66                     |                  179                             |                        52                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | laplacian                       |                      250                       |                     66                     |                  150                             |                        46                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | pixel_wise_mul                  |                      362                       |                    102                     |                  170                             |                        51                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | threshold                       |                      403                       |                    105                     |                  251                             |                        68                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   | zero function                   |                      405                       |                    103                     |                  204                             |                        66                     |
   +---------------------------------+------------------------------------------------+--------------------------------------------+--------------------------------------------------+-----------------------------------------------+
   