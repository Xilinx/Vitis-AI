function DeepRect(cfg) {
  this.selector = cfg;
  this.labels = {};
  this.labelColorIdx = 0;
  this.labelColors = ['blue', 'red', 'green', 'orange', 'purple', 'deeppink',
    'cyan', 'yellow', 'brown'];
}

//const dr = new DeepRect("#my_image_wrapper");

// Draw Rectangle
DeepRect.prototype.draw = function(x,y,w,h, label) {
  // NOTE: origin (0,0) is top-left
  let labelColor = null;

  if (label in this.labels) {
    labelColor = this.labels[label];
  } else {
    labelColor = this.labelColors[this.labelColorIdx];
    this.labelColorIdx = (this.labelColorIdx + 1) % this.labelColors.length;
    this.labels[label] = labelColor;
  }

  $('<div/>', {
    class: 'deep-rect',
    style: 'position: absolute; z-index: 100; '
         + `top: ${y}px; `
         + `left: ${x}px; `
         + `width: ${w}px; `
         + `height: ${h}px; `
         + `border: 2px solid ${labelColor}; `
         + `display: inline; `,
  }).appendTo(this.selector);

  var labelY = y + 2;
  var labelX = x + 2;
  $('<div/>', {
    class: 'deep-rect',
    style: 'position: absolute; z-index: 100; '
         + `top: ${labelY}px; `
         + `left: ${labelX}px; `
         + 'font-weight: bold; '
         + 'white-space: pre-wrap; '
         + 'background-color: rgba(0,0,0,0.5); '
         + `color: ${labelColor}; `
         + `display: inline; `,
    html: label,
  }).appendTo(this.selector);
};

DeepRect.prototype.clear = function() {
  $(this.selector + ' .deep-rect').remove();
  this.labelColorIdx = 0;
  this.labels = {};
};

//function set_resp(val){
//  console.log(val)
//  const resp = val;
//  dr.clear();
//  if (resp) {
//    for (let i = 0; i < resp.length; i++) {
//      const entry = resp[i];
//      dr.draw(entry[0], entry[1], entry[2], entry[3], entry[4]);
//    }
//  }
//}
