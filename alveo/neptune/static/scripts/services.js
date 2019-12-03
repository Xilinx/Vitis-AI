function ServicesWidget(cfg) {
  this.selector = cfg.selector;
  this.init();
}
ServicesWidget.prototype.init = function() {
  this.refreshList();
}
ServicesWidget.prototype.refreshList = function() {
  var sw = this;
  const SERVICE_STARTING = 1
  const SERVICE_STARTED = 2

  $.get("/services/list", function(response) {
    var data = JSON.parse(response);
    $(sw.selector).html("<b>Services</b>");

    for (var i=0; i < data.services.length; i++)
    {
      var serviceName = data.services[i].name;
      var row = $("<div/>", {
        class: 'row'
      });
      var label = $("<div/>", {
        class: "label col-2",
        html: serviceName
      });
      var buttonClass = 'btn-outline-secondary';
      var buttonLabel = 'Stopped';
      if (data.services[i].state == SERVICE_STARTED)
      {
        buttonClass = 'btn-success';
        buttonLabel = 'Started';
      }
      else if (data.services[i].state == SERVICE_STARTING)
      {
        buttonClass = 'btn-info';
        buttonLabel = 'Starting';
      }
      var buttonWrapper = $("<div/>", {
        class: "button col-2"
      });
      var button = $("<button/>", {
        'type': button,
        'class': "btn btn-sm " + buttonClass,
        'data-service-name': serviceName,
        'data-service-state': data.services[i].state,
        'html': buttonLabel
      });
      buttonWrapper.append(button);
      // make button send start/stop command to server
      button.click(function() {
        button.addClass("disabled");
        var serviceName = $(this).data('service-name');
        var isServiceOn = $(this).data('service-state') >= SERVICE_STARTING;
        var action = 'stop';
        if (!isServiceOn)
          action = 'start';
        var url = "/services/" + action + "?id=" + serviceName;
        $.get(url, function(response) {
          console.log(url);
        });
      });
      var url = $("<div/>", {
        class: "url col-2",
        html: data.services[i].url
      });
      var throughput = data.services[i].throughput;
      for (k in throughput)
        throughput[k] = Math.round(parseFloat(throughput[k]));
      var throughputText = "";
      if (Object.keys(throughput).length)
        throughputText = JSON.stringify(throughput, null, 2);
      var stats = $("<div/>", {
        class: "stats col text-monospace text-muted",
        style: "font-size: small",
        html: throughputText
      });
      row.append(label).append(buttonWrapper).append(url).append(stats);
      $(sw.selector).append(row);
    }
  });

  setTimeout(function() {
    sw.refreshList();
  }, 5000);
}
