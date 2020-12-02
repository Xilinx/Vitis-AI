function DeepRect(cfg) {
  this.id = cfg;
  this.labels = {};
  this.labelColorIdx = 0;
  this.labelColors = ['red', 'green', 'blue', 'orange', 'purple', 'deeppink',
    'cyan', 'yellow', 'brown'];
}

const dr = new DeepRect("my_image_wrapper");

// Draw Rectangle
DeepRect.prototype.draw = function deepRectDraw(x, y, w, h, label) {
  let labelColor = null;

  if (label in this.labels) {
    labelColor = this.labels[label];
  } else {
    labelColor = this.labelColors[this.labelColorIdx];
    this.labelColorIdx = (this.labelColorIdx + 1) % this.labelColors.length;
    this.labels[label] = labelColor;
  }

  const width = w - x;
  const height = h - y;
  $('<div/>', {
    class: 'deep-rect',
    style: 'position: absolute; '
         + `top: ${y}px; `
         + `left: ${x}px; `
         + `width: ${width}px; `
         + `height: ${height}px; `
         + `border: 2px solid ${labelColor}; `
         + `display: inline; `,
  }).appendTo('#my_image_wrapper');

  $('<div/>', {
    class: 'deep-rect',
    style: 'position: absolute; '
         + `top: ${y}px; `
         + `left: ${x}px; `
         + 'font-weight: bold; '
         + `color: ${labelColor}; `
         + `display: inline; `,
    html: label,
  }).appendTo('#my_image_wrapper');
};

DeepRect.prototype.clear = function deepRectClear() {
  $('.deep-rect').remove();
  this.labelColorIdx = 0;
  this.labels = {};
};

function set_resp(val){
  console.log(val)
  const resp = val;
  dr.clear();
  if (resp) {
    for (let i = 0; i < resp.length; i++) {
      const entry = resp[i];
      dr.draw(entry[0], entry[1], entry[2], entry[3], entry[4]);
    }
  }
}
