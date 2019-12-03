// add sum() and avg() to Array class
Array.prototype.sum = Array.prototype.sum || function() {
  return this.reduce(function(sum, a) { return sum + Number(a) }, 0);
}
Array.prototype.avg = Array.prototype.average || function() {
  return this.sum() / (this.length || 1);
}

function getUrlParam(paramName) {
  var query = window.location.search.substring(1);
  var vars = query.split('&');
  for (var i = 0; i < vars.length; i++)
  {
    var pair = vars[i].split('=');
    if (pair[0] == paramName)
      return pair[1];
   }
  return "";
}

function UI(cfg) {
  this.selector = cfg.selector;
  this.num_cores = cfg.num_cores;
  if (!this.num_cores)
    this.num_cores = 2;
  this.num_devices = cfg.num_devices;
  this.fpsSelectorPrefix = cfg.fps_selector_prefix;
  this.topsSelector = cfg.tops_selector;
  this.powerSelector = cfg.power_selector;
  this.accuracySelector = cfg.accuracy_selector;
  this.slides = cfg.slides;
  this.google = cfg.google;

  this._imgCardsClass = "img-cards";
  this._imgSpinnerClass = "img-spinner";
  this._bodyClass = "body";
  this._bodyBatchSizeId = "batch-size";
  this._bodyDeviceFpsChartId = "device-fps";
  this._bodyTimelineChartId = "timeline";

  this._imgCardsUpdateInterval = 2500; // ms
  this._imgSpinnerFPS = 20;
  this._imgSpinnerBuffer = 100;
  this._imgFPSupdateInterval = 500; // ms
  this._slidesUpdateInterval = 20000; // ms

  this._imgSpinnerUrls = []; // set by updateWithNewData()
  this._imgCardInfos = [];   // set by updateWithNewData()
  this._imgFPScounter = 0;  // computed from browser POV
  this._imgFPSfromFPGA = 0; // direct from FPGA
  this._imgLatencyfromFPGA = 0; // direct from FPGA
  this._imgCoreFPSfromFPGA = {}; // direct from FPGA
  this._imgDeviceFPSfromFPGA = {}; // direct from FPGA
  this._imgDeviceLatencyfromFPGA = {}; // direct from FPGA
  this._imgFPSfromFPGAbatchSize = 0;
  this._predictionHistory = []; // array of true/false 
  this._slidesCounter = 0;

  this._synsetLabelMap = {};
  this._imgIdx2GoldId = {};

  this.setupHTML();
  this.init();

  this._deviceFpsChart = new DeviceFpsChart({
    element_id: this._bodyDeviceFpsChartId,
    device_pfx: "device-fps-",
    num_devices: this.num_devices
  });

  this._timelineChart = new TimelineChart({
    google: this.google,
    element_id: this._bodyTimelineChartId,
    num_cores: this.num_cores
  });

  var query = window.location.search.substring(1);
  var vars = query.split('&');
  for (var i = 0; i < vars.length; i++) 
  {
    var pair = vars[i].split('=');
    if (pair[0] == 'timeline' && pair[1] == 'off')
      $("#" + this._bodyTimelineChartId
        + "-wrapper").hide();
  }
  if (getUrlParam("xdnn") == "v2")
    {
      $(".v3").hide();
      $(".v2").show();
    }
}
UI.prototype.setupHTML = function() {
  var ui = this;

  $(ui.selector).empty();

  var imgSpinnerContainer = $("<div/>", {
    class: ui._imgSpinnerClass + "-container",
    html: ""
  });
  var imgSpinner = $("<div/>", {
    class: ui._imgSpinnerClass
  });
  var imgCards = $("<div/>", {
    class: ui._imgCardsClass
  });
  var body = $("<div/>", {
    class: ui._bodyClass
  });
  var html = "";
  for (var di=0; di < ui.num_devices; di++)
    html += "<div class='wrapper'>"
      + "<canvas id='device-fps-" + di 
      + "' width='685' height='40'></canvas>"
      + "<div class='label'"
      + " id='device-fps-text-" + di 
      + "'></div></div>";
  var deviceChart = $("<div/>", {
    id: ui._bodyDeviceFpsChartId,
    html: html
  });
  var timelineChartWrapper = $("<div/>", {
    id: ui._bodyTimelineChartId + "-wrapper",
    html: "<div id='"+ui._bodyBatchSizeId+"'></div>"
      + "<div id='"+ui._bodyTimelineChartId+"'></div>"
  });

  body.append(deviceChart);
  body.append(timelineChartWrapper);
  imgSpinnerContainer.append(imgSpinner);
  $(ui.selector).append(imgSpinnerContainer);
  $(ui.selector).append(imgCards);
  $(ui.selector).append(body);
}
UI.prototype.init = function() {
  this.updateImageCards();
  this.updateImageSpinner();
  this.updateRealtimeMetrics();
  this.loadSynsetLabels();
  this.loadGoldLabels();

  //this._drawSlide(); // draw first slide
  //this.updateSlides(); // start slideshow
}
UI.prototype.loadSynsetLabels = function() {
  var ui = this;

  $.ajax({
    url: "data/synset_words.txt",
  }).done(function(data) {
    data = data.trim();
    var lines = data.split("\n");
    for (var i=0; i < lines.length; i++)
    {
      var id = lines[i].split(" ")[0];
      var labels = lines[i].substr(lines[i].indexOf(" ")+1);
      ui._synsetLabelMap[id] = labels;
    }
  });
}
UI.prototype.loadGoldLabels = function() {
  var ui = this;

  $.ajax({
    url: "data/gold.txt",
  }).done(function(data) {
    data = data.trim();
    var lines = data.split("\n");
    for (var i=0; i < lines.length; i++)
      ui._imgIdx2GoldId[i+1] = lines[i];
  });
}
UI.prototype._removeOldCards = function() {
  var ui = this;

  var cardsSelector = ui.selector 
    + " ." + ui._imgCardsClass;

  // remove old cards
  if ($(cardsSelector).children().length > 3)
  {
    //$(cardsSelector + " .card:last").remove();
    $(cardsSelector + " .card:last")
      .css('border-right', '1px solid transparent')
      .animate({
        'width': "0px",
        'opacity': 0
      }, function() {
        this.remove();
        ui._removeOldCards();
      });
  }
}
UI.prototype._getImageIdxFromPath = function(path) {
  var underscoreSplit = path.split('_');
  var numPlusSuffix = underscoreSplit[underscoreSplit.length-1];
  var idx = parseInt(numPlusSuffix.split('.')[0]);
  return idx;
}
UI.prototype.updateImageCards = function() {
  var ui = this;

  setTimeout(function() {
    requestAnimationFrame(function() {
      ui.updateImageCards();
    });

    // pick random one off the queue
    //var info = ui._imgCardInfos.pop();
    var info = ui._imgCardInfos[
      Math.floor(Math.random()
                 *ui._imgCardInfos.length)];
    if (!info)
      return;

    // flush queue for new batch
    ui._imgCardInfos = [];

    var cardsSelector = ui.selector 
      + " ." + ui._imgCardsClass;

    var card = $("<div/>", { 
      class: 'card'
    });
    var expectedLabel = $("<div/>", {
      class: 'expected-label'
    });
    var img = $("<div/>", {
      class: 'img'
    });
    var labels = $("<div/>", {
      class: 'labels'
    });

    // get expected labels
    var expectedLabels = ui._getExpectedLabels(info.path);
    var expectedLabelStr = null;
    if (expectedLabels)
    {
      expectedLabelStr = expectedLabels[0];
      expectedLabel.html(expectedLabelStr);
    }

    // get our predictions, check results
    var foundMatch = false;
    for (var p=0; p < info.predictions.length; p++)
    {
      var line = info.predictions[p];
      var els = line.split(" ");
      var val = parseInt(parseFloat(els[0]) * 100);
      var lstring = "";
      var id = els[1];
      var tags = els.slice(2, els.length)
        .join(" ").split(",");
      for (var ei=0; ei < 1/*tags.length*/; ei++)
      {
        var tag = tags[ei].replace(',', '');
        lstring += tag + " ";
      }

      if (expectedLabelStr && !foundMatch)
      {
        var origString = lstring;
        lstring = lstring.replace(expectedLabelStr,
          "<span class='match'>" + expectedLabelStr 
          + "</span>");
        
        if (lstring != origString)
          foundMatch = true;
      }

      var label = $("<div/>", {
        class: 'label',
        html: val + "% " + lstring
      });
      labels.append(label);
    }

    // set the card's image
    img.css('background-image', 
      'url(' + info.path + ')');

    card.append(expectedLabel);
    card.append(img);
    card.append(labels);
    $(cardsSelector).prepend(card);

    var origWidth = card.css('width');
    card
      .css('width', '0px')
      .css('border-left', '1px solid transparent')
      .animate({
        'width': origWidth
      }, function() {
        $(this).css('border', '1px solid #aaa');
      });

    ui._removeOldCards();
  }, ui._imgCardsUpdateInterval);
}
UI.prototype.updateImageSpinner = function() {
  var ui = this;

  setTimeout(function() {
    requestAnimationFrame(function() {
      ui.updateImageSpinner();
    });

    //var url = ui._imgSpinnerUrls.shift();
    var url = ui._imgSpinnerUrls[
      Math.floor(Math.random()
                 *ui._imgSpinnerUrls.length)];
    if (!url)
      return;

    var spinnerSelector = ui.selector 
      + " ." + ui._imgSpinnerClass;

    // accumulate a stack of images to ensure we see them
    if ($(spinnerSelector).children().length
      > ui._imgSpinnerBuffer)
    {
      // after the stack is full, just rotate within stack
      // instead of using new images (to conserve bandwidth)
      var topDiv = $(spinnerSelector).find("div:first");
      $(spinnerSelector).append(topDiv);
    }
    else
    {
      // stack is not full yet -- add to stack
      var div = $("<div/>", { });
      div.css('background-image', 'url(' + url + ')');
      $(spinnerSelector).append(div);
    }
  }, 1000 / ui._imgSpinnerFPS);
}
UI.prototype.updateRealtimeMetrics = function() {
  var ui = this;

  setTimeout(function() {
    requestAnimationFrame(function() {
      ui.updateRealtimeMetrics();

      var period = ui._imgFPSupdateInterval / 1000;
      var fps = ui._imgFPScounter / period;

      if (ui._imgFPSfromFPGA)
      {
        // override with FPGA measurement
        fps = ui._imgFPSfromFPGA;
      }

      var fps16 = fps.toFixed(0);
      var fps8 = 0;
      if(getUrlParam("xdnn")=="v2")
        fps8 = (fps*2).toFixed(0); 
      else
        fps8 = (fps).toFixed(0); 
      ui._imgFPScounter = 0; // reset
      var topsFactor = 4.272 / 1000; // 267*8000*2
      // 3.3 for F1
      var tops = (fps8 * topsFactor / 2).toFixed(2);

      $(ui.fpsSelectorPrefix + "16").html(fps16);
      $(ui.fpsSelectorPrefix + "8").html(fps8);
      if (ui._imgLatencyfromFPGA)
        {
          var latencyFactor = 1.18/0.96;
          var latency=(ui._imgLatencyfromFPGA*latencyFactor).toFixed(2);
          var peakLatency = ui._imgLatencyfromFPGA.toFixed(2);
          $(ui.fpsSelectorPrefix + "latency").html(latency);
          $(ui.fpsSelectorPrefix + "peakLatency").html(peakLatency);

        }
      $(ui.topsSelector).html(tops);

      if (fps8 > 0)
      {
        //$(ui.powerSelector).html(
        //  (42 + (Math.random()*2)).toFixed(1));
        $(ui.powerSelector).html(43);
      }

      // update accuracy %
      var numCorrect = 0;
      for (var i=0; 
        i < ui._predictionHistory.length; i++)
        if (ui._predictionHistory[i])
          numCorrect++;
      var pct = numCorrect * 100
        / ui._predictionHistory.length;
      if (ui._predictionHistory.length > 0)
        $(ui.accuracySelector).html(pct.toFixed(0) + "%");
    });
  }, ui._imgFPSupdateInterval);
}
UI.prototype.updateSlides = function() {
  var ui = this;

  setTimeout(function() {
    requestAnimationFrame(function() {
      ui.updateSlides();

      ui._drawSlide();
    });
  }, ui._slidesUpdateInterval);
}
UI.prototype._drawSlide = function() {
  var ui = this;
  var html = "<img src='" 
    + ui.slides[ui._slidesCounter]
    + "'/>";
  $(ui.selector + " ." + ui._bodyClass).html(html);

  ui._slidesCounter 
    = ((ui._slidesCounter+1) % ui.slides.length);
}
UI.prototype.updateProfileData = function(data) {
  var ui = this;

  data = data.trim();
  var keyValues = data.split(",");
  var profMap = {};

  for (var i=0; i < keyValues.length; i++)
  {
    var kv = keyValues[i].split(" ");
    profMap[kv[0]] = parseFloat(kv[1]); // ns
  } 

  // update FPS counter
  ui._imgFPSfromFPGAbatchSize 
    = profMap.b * ui.num_cores * ui.num_devices;
  var extraExecBarLabel = "";
  if (ui._imgFPSfromFPGAbatchSize)
  {
    var delta =  ((profMap.k_end - profMap.k_start) / 1000000000);

    var fps = ui._imgFPSfromFPGAbatchSize / delta;
    ui._imgCoreFPSfromFPGA[profMap.h + "/" + profMap.k] = fps;
    ui._imgDeviceFPSfromFPGA[profMap.h] = fps;
    ui._imgDeviceLatencyfromFPGA[profMap.h] = delta;
    var totalFps = 0;
    var numCores = 0;
    for (var k in ui._imgCoreFPSfromFPGA) 
    {
      totalFps += ui._imgCoreFPSfromFPGA[k];
      numCores++;
    }
    ui._imgLatencyfromFPGA = delta*1000;
    ui._imgFPSfromFPGA = totalFps / numCores;
    //extraExecBarLabel += " / " + parseInt(fps) + " fps";

    // update device fps ticker
    if (profMap.h !== undefined)
    {
      var deviceFps = fps / ui.num_devices;
      var val = 0;
      if(getUrlParam("xdnn")=="v2")
        val = parseInt(deviceFps+0.5) * 2; // x2 because 8-bit
      else
         val = parseInt(deviceFps+0.5);
      ui._deviceFpsChart.append(parseInt(profMap.h), val);
      $("#device-fps-text-" + profMap.h).html(val+" fps");
    }
  }
  $("#"+ui._bodyBatchSizeId).html(
    "batch size per core: " 
    + ui._imgFPSfromFPGAbatchSize/ui.num_cores);

  // update timeline chart rows
  if (profMap.h === undefined || profMap.h == 0)
  {
    var timeLimit = 50000;
    var skipUpdate = false;
    
    var writeEndTime = parseFloat(profMap.w_end - profMap.w_start) / 1000.;
    var kernelEndTime = parseFloat(profMap.k_end - profMap.k_start) / 1000.;
    var readEndTime = parseFloat(profMap.r_end - profMap.r_start) / 1000.;

    if(!(getUrlParam("xdnn")=="v2") && (writeEndTime > timeLimit || kernelEndTime > timeLimit || readEndTime > timeLimit))
      skipUpdate = true;
    
    if(!skipUpdate)
      {
        ui._timelineChart.updateRow("Core_" + profMap.k + " - Write", 
      profMap.w_start, profMap.w_end, null, false);
        ui._timelineChart.updateRow("Core_" + profMap.k + " - Exec", 
      profMap.k_start, profMap.k_end, extraExecBarLabel, false);
        ui._timelineChart.updateRow("Core_" + profMap.k + " - Read", 
      profMap.r_start, profMap.r_end, null, true);
      }
  }
}
UI.prototype._getExpectedLabels = function(path) {
  var ui = this;
  var imgIdx = ui._getImageIdxFromPath(path);
  var goldId = ui._imgIdx2GoldId[imgIdx];
  var expectedLabels = ui._synsetLabelMap[goldId];
  if (expectedLabels)
    expectedLabels = expectedLabels.split(',');
  else
    expectedLabels = [];

  return expectedLabels;
}
UI.prototype._isTop5Correct = function(info) {
  var ui = this;

  var path = info.path;
  var predictions = info.predictions;
  var expected = ui._getExpectedLabels(path);

  for (var p=0; p < predictions.length; p++)
  {
    var line = info.predictions[p];
    var firstWords = line.split(",")[0];
    var secondSpace = firstWords.indexOf(' ', 
      firstWords.indexOf(' ')+1);
    var firstTag = firstWords.substr(secondSpace+1);
  
    if ($.inArray(firstTag, expected) >= 0)
      return true;
  }

  return false;
}
UI.prototype.updateWithNewData = function(data) {
  var ui = this;

  // parse xteng syntax
  var lines = data.match(/[^\r\n]+/g);
  var results = [];

  for (var i=0; i < lines.length; i++)
  {
    var rowsPerPrediction = 6;

    var idx = parseInt(i / rowsPerPrediction);
    var offset = i % rowsPerPrediction;
    if (offset == 0)
    {
      results.push({});
      results[idx].path = lines[i];
    }
    else
    {
      var line = lines[i].trim();
      if (offset == 1)
        results[idx].predictions = [];
      results[idx].predictions.push(line);
    }
  }

  // clear backlog of images in spinner queue
  ui._imgSpinnerUrls = [];

  // process results
  for (var i=0; i < results.length; i++)
  {
    if (!results[i].predictions)
      continue;

    // update image spinner, cards and FPS counter
    ui._imgSpinnerUrls.push(results[i].path);
    ui._imgCardInfos.push(results[i]);
    ui._imgFPScounter++;

    // update prediction history
    if (ui._isTop5Correct(results[i]))
      ui._predictionHistory.push(true);
    else
      ui._predictionHistory.push(false);
    if (ui._predictionHistory.length > 128)
      ui._predictionHistory.shift();
  }
}

/************************
 * FPS chart
 * (http://smoothiecharts.org)
 ************************/
function DeviceFpsChart(cfg) {
  this.element_id = cfg.element_id;
  this.device_pfx = cfg.device_pfx;
  this.num_devices = cfg.num_devices;

  this.data = [];
  this.charts = [];
  for (var i=0; i < this.num_devices; i++)
  {
    var ts = new TimeSeries();
    this.data.push(ts);
    var chart = new SmoothieChart({
      millisPerPixel: 20, 
      grid: { 
        strokeStyle: '#999999', 
        lineWidth: 1, 
        millisPerLine: 1000, 
        verticalSections: 4 
      },
      labels: { disabled: true, fontSize: 12 },
      minValue: 1000
    });
    var options = {
      strokeStyle: 'rgba(0, 255, 0, 1)', 
      fillStyle: 'rgba(0, 255, 0, 0.2)', 
      lineWidth: 3 
    };
    chart.addTimeSeries(ts, options);
    chart.streamTo(document.getElementById(
      this.device_pfx + i), 1000);
    this.charts.push(chart);
  }
}
DeviceFpsChart.prototype.append
  = function(deviceId, fps) {
  this.data[deviceId].append(
    new Date().getTime(), fps);
}

/************************
 * Google Timeline chart
 ************************/
function TimelineChart(cfg) {
  this.google = cfg.google;
  this.elementId = cfg.element_id;
  this.num_cores = cfg.num_cores;

  this._chart = null;
  this._dataTable = null;
  this._event2RowMap = {};
  this._runtimeLog = [1]; // last N runtimes

  var tc = this;
  this.google.charts.load('current', {'packages':['timeline']});
  this.google.charts.setOnLoadCallback(function() {
    tc.init()
  });
}
TimelineChart.prototype.init = function() {
  var tc = this;
  var container = document.getElementById(tc.elementId); 
  tc._chart = new tc.google.visualization.Timeline(container);
  tc._dataTable = new tc.google.visualization.DataTable();

  tc._dataTable.addColumn({ type: 'string', id: 'Event' });
  tc._dataTable.addColumn({ type: 'string', id: 'Label' });
  tc._dataTable.addColumn({ type: 'date', id: 'Start' });
  tc._dataTable.addColumn({ type: 'date', id: 'End' });

  var eventNames = [];
  for (var i=tc.num_cores-1; i >= 0; i--)
  {
    eventNames.push("Core_" + i + " - Read");
    eventNames.push("Core_" + i + " - Exec");
    eventNames.push("Core_" + i + " - Write");
  }

  var rows = [];
  for (var i=0; i < eventNames.length; i++)
  {
    var name = eventNames[i];
    tc._event2RowMap[name] = i;

    // set default row
    rows.push([ name, "", new Date(0), new Date(1) ]);
  }

  tc._dataTable.addRows(rows);

  tc.drawChart();
}
TimelineChart.prototype.drawChart = function() {
  var tc = this;
  
  if (!tc._chart)
    return; // chart not yet ready

  tc._chart.clearChart();

  var avgRuntime = tc._runtimeLog.avg();
  var maxRuntime = Math.max.apply(null, tc._runtimeLog);
  //var chartMax = (maxRuntime - avgRuntime) / 2;
  //chartMax += avgRuntime;
  var chartMax = maxRuntime;

  var chartOptions = {
    colors: ['#00aa00', '#ff0000', '#0088ff'],
    backgroundColor: '#eee',
    enableInteractivity: false,
    timeline: { 
      groupByRowLabel: false,
      colorByRowLabel: true 
    },
    hAxis: {
      minValue: new Date(0),
      maxValue: new Date(chartMax)
    }
  };
  tc._chart.draw(tc._dataTable, chartOptions);
}
TimelineChart.prototype.updateRow 
  = function(eventName, startTime, endTime, appendLabel, redraw) {
  var tc = this;

  if (!tc._chart)
    return; // chart not yet ready

  startTime = parseFloat(startTime) / 1000.;
  endTime = parseFloat(endTime) / 1000.;
  
  if (startTime > 10000)
    return; // ANDBG hack - skip outliers

  var barLabel = parseFloat((endTime - startTime)/1000)
    .toFixed(2).toString() + " ms";
  if (appendLabel)
    barLabel += appendLabel;

  var rowIdx = tc._event2RowMap[eventName];
  tc._dataTable.setValue(rowIdx, 1, barLabel);
  tc._dataTable.setValue(rowIdx, 2, new Date(startTime));
  tc._dataTable.setValue(rowIdx, 3, new Date(endTime));

  if (eventName.indexOf("Read") !== -1)
  {
    // save end time of this batch to determine 
    // chart's hAxis range max
    tc._runtimeLog.push(endTime);

    if (tc._runtimeLog.length > 30)
      tc._runtimeLog.shift();
  }

  if (redraw)
    tc.drawChart();
}

/************************
 * Websockets Manager
 ************************/
function WebSocketMgr(cfg) {
  this.url = cfg.url;
  this.callback = cfg.callback;
  this.profDataCallback = cfg.profdata_callback;

  this.wsBatches = [];
  this.ws = null;

  this.init();
}
WebSocketMgr.prototype.init = function() {
  var wsMgr = this;
  var ws = this.ws;
  ws = new WebSocket(this.url);

  ws.onopen = function() {
    console.log("[WebSocketMgr] connected to "
      + wsMgr.url);
    function schedule(i) {
      setTimeout(function() {
        ws.send('Hello from the client! (i=' + i + ')');
        schedule((i + 1) % 1000);
      }, 10000);
    };
    schedule(1);
  };

  ws.onmessage = function(evt)
  {
    var message = evt.data;
    var obj = JSON.parse(message);

    if (obj.topic == "xmlrt")
    {
      if (wsMgr.profDataCallback)
        wsMgr.profDataCallback(obj.data);
      return;
    }

    wsMgr.callback(obj.data);
  };

  ws.onclose = function()
  {
    console.log("[WebSocketMgr] disconnected from "
      + wsMgr.url + ", trying to reconnect in 1 second...");
    setTimeout(function() {
      wsMgr.init();
    }, 2000);
  };
}
