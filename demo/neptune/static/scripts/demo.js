function DemoWidget(cfg) {
  this.selector = cfg.selector;
  this.wsport = cfg.wsport;
  this.uioff = cfg.uioff;

  this._serviceList = [];
  this._clientId = 0;
  if (cfg.client_id)
    this._clientId = cfg.client_id;
  this._drawbox = null;

  this.init();
}
DemoWidget.prototype.init = function() {
  var form = $("<div/>", {
    class: "form-group",
    html: "<b>Demo</b>"
  });
  var serviceList = $("<div/>", {
    class: "service-list"
  });
  var input = $("<input/>", {
    type: "text",
    class: "form-control input-url",
    placeholder: "Enter image URLs"
  });
  var jsonResponse = $("<div/>", {
    class: "response text-monospace",
    style: "white-space: pre-wrap; line-height: 14px; font-size: 14px"
  });
  var submit = $("<button/>", {
    type: "submit",
    class: "btn btn-primary",
    html: "Submit"
  });

  var canvas = $("<div/>", {
    class: "response-canvas text-monospace",
    style: "white-space: pre-wrap; line-height: 14px; font-size: 14px"
  });
  {
    var row1 = $("<div/>", { class: 'row' });
    var previewWrapper = $("<div/>", { class: 'col' });
    var preview = $("<div/>", { class: 'preview', style: 'position: relative' });
    var img = $("<img/>", { });
    var row2 = $("<div/>", { class: 'row' });
    var json = $("<div/>", { class: 'col json', style: 'display: none' });

    preview.append(img);
    previewWrapper.append(preview);
    row1.append(previewWrapper);
    row2.append(json);
    canvas.append(row1).append(row2);
  }
  this._drawbox = new DeepRect(this.selector + " .response-canvas .preview");

  form.append(serviceList).append(input).append(jsonResponse).append(submit);
  $(this.selector).append(form).append(canvas);
  if (this.uioff)
    form.hide();

  this.bindEvents();
  this.refreshServiceList();
  this.connectServerStream();
}
DemoWidget.prototype.submit = function(inputUrl, serviceNames, keepAlive) {
  var dw = this;

  $(dw.selector + " .input-url").val(inputUrl);
  var serviceFound = false;
  for (var i=0; i < serviceNames.length; i++)
  {
    var checkbox = $(dw.selector + " .form-check .demo-" + serviceNames[i])
    if (checkbox.length)
      serviceFound = true;
    checkbox.prop("disabled", false).prop("checked", true);
    checkbox.data('keepalive', keepAlive);
  }

  if (!serviceFound)
  {
    // service list not populated yet, try again after a delay
    setTimeout(function() {
      dw.submit(inputUrl, serviceNames, keepAlive);
    }, 1000);
    return;
  }

  $(this.selector + " button[type=submit]").trigger("click");
}
function toDataUrl(url, callback) {
    var xhr = new XMLHttpRequest();
    xhr.onload = function() {
        var reader = new FileReader();
        reader.onloadend = function() {
            callback(reader.result);
        }
        reader.readAsDataURL(xhr.response);
    };
    xhr.open('GET', url);
    xhr.responseType = 'blob';
    xhr.send();
}
DemoWidget.prototype.drawResponseOnCanvas = function(resp) {
  var dw = this;
  var canvas = $(dw.selector + " .response-canvas");

  var imgData = "data:image/jpeg;base64," + resp.img;
  delete resp.img;
  //console.log(resp);

  // update JSON
  canvas.find(".json").html(JSON.stringify(resp, null, 4));

  // update image
  var tmpImg = new Image();
  tmpImg.src = imgData;
  tmpImg.onload = function() {
    var img = canvas.find("img");
    img.attr('src', imgData);

    img.data('orig-width', resp.image_width);
    img.data('orig-height', resp.image_height);
    img.data('aspect-ratio', resp.image_width / resp.image_height);

    // upsize image to square
    if (tmpImg.width <= tmpImg.height)
    {
      img.width(tmpImg.height);
      img.height(tmpImg.height);
      img.data('scale-x', tmpImg.height/tmpImg.width);
      img.data('scale-y', 1);
    }
    else
    {
      img.width(tmpImg.width);
      img.height(tmpImg.width);
      img.data('scale-x', 1);
      img.data('scale-y', tmpImg.width/tmpImg.height);
    }

    // reshape image to original aspect ratio
    if (img.data('aspect-ratio') >= 1) // landscape
    {
      var scaleY = 1. / img.data('aspect-ratio');
      img.height(img.width() * scaleY);
      img.data('scale-y', img.data('scale-y')*scaleY);
    }
    else // portrait
    {
      var scaleX = img.data('aspect-ratio');
      img.width(img.height() * scaleX);
      img.data('scale-x', img.data('scale-x')*scaleX);
    }

    // expand/shrink image width to parent container's width
    if (img.width() != img.parent().width())
    {
      var scale = img.parent().width() / img.width();
      img.width(img.width() * scale);
      img.height(img.height() * scale);
      img.data('scale-x', img.data('scale-x')*scale);
      img.data('scale-y', img.data('scale-y')*scale);
    }

    if (resp.boxes)
    {
      dw._drawbox.clear();
      var boxes = resp.boxes;
      for (var i=0; i < boxes.length; i++)
      {
        const entry = boxes[i];
        var w = entry[2] - entry[0];
        var h = entry[3] - entry[1];
        var x = entry[0];
        var y = entry[1];

        dw._drawbox.draw(
          x*img.data('scale-x'), y*img.data('scale-y'),
          w*img.data('scale-x'), h*img.data('scale-y'), entry[4]);
      }
    }
  };
}
DemoWidget.prototype.connectServerStream = function() {
  var dw = this;
  var hostname = window.location.hostname;
  dw._wsMgr = new WebsocketMgr({
    url: "ws://" + hostname + ":" + dw.wsport + "/",
    callbacks: {
      'id': function(data) {
        console.log("Demo widget client ID: " + data);

        if (dw._clientId == 0)
        {
          // first time getting an ID, use it
          dw._clientId = data;
        }
        else
        {
          // already had an ID, tell server to use our orig ID
          dw._wsMgr.send(JSON.stringify({
            'topic': 'update_id',
            'id': dw._clientId
          }));
        }

      },
      'callback': function(data) {
        var resp = JSON.parse(data);

        dw.drawResponseOnCanvas(resp);
      }
    }
  });
}
DemoWidget.prototype.bindEvents = function() {
  var dw = this;
  var responseBox = $(dw.selector + " .response");

  var submitButton = $(this.selector + " button[type=submit]");
  submitButton.click(function() {
    responseBox.empty();

    var inputUrl = $(dw.selector + " .input-url").val();
    if (!inputUrl)
      return;

    var hasKeepAlive = false;
    $(dw.selector + " .service-list .form-check-input").each(function() {
      var el = $(this);
      if (!el.prop('checked'))
        return;

      if (el.data('keepalive'))
        hasKeepAlive = true;

      if (dw._clientId == 0)
        return; // haven't received ID from server yet

      const serviceUrl = el.data('url');
      var url = serviceUrl + "?url=" + inputUrl + "&video_type=live&dtype=uint8"
        + "&callback_id=" + dw._clientId;
      console.log(url);

      el.data('last-input-url-submitted', inputUrl);
      $.get(url, function(resp) {
        var data = JSON.parse(resp);
        responseBox.html(responseBox.html()
          + "\n" + JSON.stringify(data, null, 4));
      });
    });

    if (submitButton.data('keepalive-timer'))
    {
      // cancel any existing keepalive functions
      clearTimeout(submitButton.data('keepalive-timer'));
      submitButton.data('keepalive-timer', null);
    }

    if (hasKeepAlive)
    {
      // auto-click the button again
      var timer = setTimeout(function() {
        submitButton.trigger("click");
      }, 5000);

      submitButton.data('keepalive-timer', timer);
    }
  });
}
DemoWidget.prototype.refreshServiceList = function() {
  var dw = this;

  $.get("/services/list", function(response) {
    var data = JSON.parse(response);

    var list = $(dw.selector + " .service-list");
    for (var i=0; i < data.services.length; i++)
    {
      var chkExists = $(dw.selector + " .form-check .demo-"
        + data.services[i].name);
      if (chkExists.length)
      {
        // existing entry -- update state if necessary
        if (data.services[i].state)
          chkExists.removeAttr("disabled");
        else
        {
          chkExists.prop("checked", false);
          chkExists.prop("disabled", true);
        }
        continue;
      }

      // new entry -- add to list
      var row = $("<div/>", {
        class: 'form-check'
      });
      var box = $("<input/>", {
        class: 'form-check-input demo-' + data.services[i].name,
        type: 'checkbox'
      });
      box.data('url', data.services[i].url);
      if (!data.services[i].state)
        box.prop('disabled', true);
      var label = $("<label/>", {
        class: 'form-check-label',
        for: 'demo-' + data.services[i].name,
        html: data.services[i].name
      });

      row.append(box).append(label);
      list.append(row);
    }
  });

  setTimeout(function() {
    dw.refreshServiceList();
  }, 5000);
}
