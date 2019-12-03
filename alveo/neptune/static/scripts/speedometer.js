function SpeedoStats() {
  this.vals = [];
  this.vrange = null;
  this.winsize = 10;
}
SpeedoStats.prototype.add = function(v) {
  this.vals.push(v);
  this.vals = this.vals.slice(-this.winsize);
  if (this.vrange == null)
    this.vrange = [v, v];
  else
    this.vrange = [Math.min(v, this.vrange[0]), 
                   Math.max(v, this.vrange[1])];
}
SpeedoStats.prototype.mean = function() {
  if (this.vals.length == 0)
    return 0.

  return this.vals.reduce((a, b) => a + b, 0) / this.vals.length;
}
SpeedoStats.prototype.min = function(dynamic) {
  if (this.vals.length == 0)
    return 0.

  if (dynamic)
    return Math.min(...this.vals);

  return this.vrange[0];
}
SpeedoStats.prototype.max = function(dynamic) {
  if (this.vrange == null)
    return 0.;

  if (dynamic)
    return Math.max(...this.vals);

  return this.vrange[1];
}
SpeedoStats.prototype.n = function() {
  return this.vals.length
}

function Speedometer(cfg) {
  this.selector = cfg.selector;

  this.rows = ["quant", "prep", "ddr_wr", "submit", 
               "hw_counter_0", "hw_counter_1", "ddr_rd", "post"];

  this.stats = {
    "input": new SpeedoStats(),
    "input_id": new SpeedoStats(),
    "pe": new SpeedoStats(),
    "total": new SpeedoStats()
  };
  for (var ri=0; ri < this.rows.length; ri++)
    this.stats[this.rows[ri]] = new SpeedoStats();

  this.lastScreenUpdate = Date.now();

  this.initLayout();
}
Speedometer.prototype.initLayout = function() {
  $(this.selector).append($("<div/>", {
    class: "title",
    html: "<b>FPGA pipeline report</b>"
  }));
  for (var ri=0; ri < this.rows.length; ri++)
  {
    var row = $("<div/>", {
      class: 'row mb-1'
    });
    var label = $("<div/>", {
      class: this.rows[ri]+"-label col-1 mr-2",
      html: this.rows[ri]
    });
    var val = $("<div/>", {
      class: this.rows[ri]+"-val col-1",
      html: ""
    });
    var barFrame = $("<div/>", {
      class: this.rows[ri]+"-bar-frame col-lg-4 col-md-10",
      style: "position: relative"
    });
    var bar = $("<div/>", {
      class: this.rows[ri]+"-bar",
      style: "position: absolute; left: 0; height: 100%; "
             + "background-color: #00CED1;"
    });
    barFrame.append(bar);
    row.append(label).append(val).append(barFrame);

    $(this.selector).append(row);
  }
  $(this.selector).append($("<div/>", {
    class: "input-rate",
    html: "<div>Input rate: <span class='val'>0</span> images/s</div>"
  }));
  $(this.selector).append($("<div/>", {
    class: "max-throughput",
    html: "<div>Max pipeline throughput: <span class='val1'>0</span> images/s "
      + " with <span class='val2'>0</span> PEs (pre-/post-processing not included)"
  }));
  $(this.selector).append($("<div/>", {
    class: "pipe-util",
    html: "<div>Pipeline utilization: <span class='val'>0</span>%"
  }));
  $(this.selector).append($("<div/>", {
    class: "latency",
    html: "<div>Pipeline latency: <span class='val'>0</span> ms"
  }));
}
Speedometer.prototype.update = function(data) {
  this.stats["input"].add(parseFloat(data['t']));
  this.stats["input_id"].add(parseFloat(data['id']));
  this.stats["pe"].add(parseFloat(data['pe']));
  this.stats["total"].add(parseFloat(data['total']));
  var max = 0;
  for (var i=0; i < this.rows.length; i++)
  {
    var key = this.rows[i];
    this.stats[key].add(parseFloat(data[key]));
    max = Math.max(this.stats[key].max(true), max);
  }

  const now = Date.now();
  // ANDBG if (now - this.lastScreenUpdate < 100)
  // ANDBG   return; // chill out and don't refresh page too much
  this.lastScreenUpdate = now;

  for (var key in this.stats)
  {
    var val = this.stats[key].mean();
    var min = this.stats[key].min();
    var pct = Math.min(100, val / max * 100);
    $(this.selector + " ." + key + "-bar").css('width', pct + "%");
    $(this.selector + " ." + key + "-val").text(val.toFixed(2) + " ms");
  }

  // compute input rate
  var tmin = this.stats["input"].min(true);
  var tmax = this.stats["input"].max(true);
  var period = (tmax - tmin) / 1000.;
  var idmin = this.stats["input_id"].min(true);
  var idmax = this.stats["input_id"].max(true);
  var n = idmax - idmin;
  var inputRate = n / period;
  $(this.selector + " .input-rate .val").text(inputRate.toFixed(2));

  // compute max pipeline throughput
  var numPE = this.stats["pe"].max() + 1;
  var pipeRate = numPE * 1000 / max;
  $(this.selector + " .max-throughput .val1").text(pipeRate.toFixed(2));
  $(this.selector + " .max-throughput .val2").text(numPE);

  // compute pipe
  var efficiency = Math.min(inputRate / pipeRate * 100., 100.);
  $(this.selector + " .pipe-util .val").text(efficiency.toFixed(2));

  // compute latency
  var latency = this.stats["total"].min(true);
  $(this.selector + " .latency .val").text(latency.toFixed(2));
}
