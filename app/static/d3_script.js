// constants

const color = [
  "rgb(209, 238, 234)",
  "rgb(168, 219, 217)",
  "rgb(133, 196, 201)",
  "rgb(104, 171, 184)",
  "rgb(79, 144, 166)",
  "rgb(59, 115, 143)",
  "rgb(42, 86, 116)",
];

const tile = d3.treemapSquarify;

var global_selecteds = [];

let isCtrlPressed = false;
let firstClicked = true;

//functions

function clearSelection(svg) {
  svg.selectAll("rect").attr("selected", false);
}

function toggleSelection(svg, name) {
  svg
    .selectAll("rect")
    .filter(function () {
      return d3.select(this).attr("name") === name;
    })
    .attr("selected", function () {
      const curValue = d3.select(this).attr("selected") === "true";
      return !curValue;
    });
}

function updateSelections(svg, flags) {
  svg.selectAll("rect").each(function () {
    d3.select(this).attr("selected", flags[d3.select(this).attr("name")]);
  });
}

function highlightSelections(svg, flags) {
  svg
    .selectAll("rect")
    .filter(function () {
      return flags[d3.select(this).attr("name")];
    })
    .attr("style", "stroke-width:3;stroke:yellow");

  svg
    .selectAll("rect")
    .filter(function () {
      return !flags[d3.select(this).attr("name")];
    })
    .attr("style", "none");
}

function handleClick(event, svg) {
  const name = event.target.getAttribute("name");

  if (!isCtrlPressed) {
    clearSelection(svg);
  }

  toggleSelection(svg, name);

  const flags = get_all_flags(svg);

  updateSelections(svg, flags);
  highlightSelections(svg, flags);

  if (!isCtrlPressed) {
    plotlySubmit();
  }
}

function simulateClick(element) {
  const clickEvent = new MouseEvent("click", {
    view: window,
    bubbles: true,
    cancelable: true,
  });
  element.dispatchEvent(clickEvent);
}

function interpolatePoints(start, end, numPoints) {
  var scale = d3
    .scaleLinear()
    .domain([0, numPoints - 1])
    .range([start, end]);

  var points = d3.range(numPoints).map(scale);
  return points;
}

function Legend(
  color,
  {
    title,
    tickSize = 6,
    width = 320,
    height = 44 + tickSize,
    marginTop = 18,
    marginRight = 0,
    marginBottom = 16 + tickSize,
    marginLeft = 0,
    ticks = width / 64,
    tickFormat,
    tickValues,
  } = {}
) {
  function ramp(color, n = 256) {
    const canvas = document.createElement("canvas");
    canvas.width = n;
    canvas.height = 1;
    const context = canvas.getContext("2d");
    for (let i = 0; i < n; ++i) {
      context.fillStyle = color(i / (n - 1));
      context.fillRect(i, 0, 1, 1);
    }
    return canvas;
  }

  const svg = d3
    .create("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", [0, 0, width, height])
    .style("overflow", "visible")
    .style("display", "block");

  let tickAdjust = (g) =>
    g.selectAll(".tick line").attr("y1", marginTop + marginBottom - height);
  let x;

  // Continuous
  if (color.interpolate) {
    const n = Math.min(color.domain().length, color.range().length);

    x = color
      .copy()
      .rangeRound(
        d3.quantize(d3.interpolate(marginLeft, width - marginRight), n)
      );

    svg
      .append("image")
      .attr("x", marginLeft)
      .attr("y", marginTop)
      .attr("width", width - marginLeft - marginRight)
      .attr("height", height - marginTop - marginBottom)
      .attr("preserveAspectRatio", "none")
      .attr(
        "xlink:href",
        ramp(
          color.copy().domain(d3.quantize(d3.interpolate(0, 1), n))
        ).toDataURL()
      );
  }

  // Sequential
  else if (color.interpolator) {
    x = Object.assign(
      color
        .copy()
        .interpolator(d3.interpolateRound(marginLeft, width - marginRight)),
      {
        range() {
          return [marginLeft, width - marginRight];
        },
      }
    );

    svg
      .append("image")
      .attr("x", marginLeft)
      .attr("y", marginTop)
      .attr("width", width - marginLeft - marginRight)
      .attr("height", height - marginTop - marginBottom)
      .attr("preserveAspectRatio", "none")
      .attr("xlink:href", ramp(color.interpolator()).toDataURL());

    if (!x.ticks) {
      if (tickValues === undefined) {
        const n = Math.round(ticks + 1);
        tickValues = d3
          .range(n)
          .map((i) => d3.quantile(color.domain(), i / (n - 1)));
      }
      if (typeof tickFormat !== "function") {
        tickFormat = d3.format(tickFormat === undefined ? ",f" : tickFormat);
      }
    }
  }

  // Threshold
  else if (color.invertExtent) {
    const thresholds = color.thresholds
      ? color.thresholds() // scaleQuantize
      : color.quantiles
      ? color.quantiles() // scaleQuantile
      : color.domain(); // scaleThreshold

    const thresholdFormat =
      tickFormat === undefined
        ? (d) => d
        : typeof tickFormat === "string"
        ? d3.format(tickFormat)
        : tickFormat;

    x = d3
      .scaleLinear()
      .domain([-1, color.range().length - 1])
      .rangeRound([marginLeft, width - marginRight]);

    svg
      .append("g")
      .selectAll("rect")
      .data(color.range())
      .join("rect")
      .attr("x", (d, i) => x(i - 1))
      .attr("y", marginTop)
      .attr("width", (d, i) => x(i) - x(i - 1))
      .attr("height", height - marginTop - marginBottom)
      .attr("fill", (d) => d);

    tickValues = d3.range(thresholds.length);
    tickFormat = (i) => thresholdFormat(thresholds[i], i);
  } else {
    x = d3
      .scaleBand()
      .domain(color.domain())
      .rangeRound([marginLeft, width - marginRight]);

    svg
      .append("g")
      .selectAll("rect")
      .data(color.domain())
      .join("rect")
      .attr("x", x)
      .attr("y", marginTop)
      .attr("width", Math.max(0, x.bandwidth() - 1))
      .attr("height", height - marginTop - marginBottom)
      .attr("fill", color);

    tickAdjust = () => {};
  }

  svg
    .append("g")
    .attr("transform", `translate(0,${height - marginBottom})`)
    .call(
      d3
        .axisBottom(x)
        .ticks(ticks, typeof tickFormat === "string" ? tickFormat : undefined)
        .tickFormat(typeof tickFormat === "function" ? tickFormat : undefined)
        .tickSize(tickSize)
        .tickValues(tickValues)
    )
    .call(tickAdjust)
    .call((g) => g.select(".domain").remove())
    .call((g) =>
      g
        .append("text")
        .attr("x", marginLeft)
        .attr("y", marginTop + marginBottom - height - 6)
        .attr("fill", "currentColor")
        .attr("text-anchor", "start")
        .attr("font-weight", "bold")
        .attr("class", "title")
        .text(title)
    );

  return svg.node();
}

function make_legend(data, color, width) {
  var root = root_from_data(data);
  var persistences = [];
  root.each((d) => {
    if (d.data.persistence != "nan") persistences.push(d.data.persistence);
  });
  var min = d3.extent(persistences)[0];
  var max = d3.extent(persistences)[1];

  return Legend(
    d3.scaleLinear(interpolatePoints(min, max, 7), color, (width = width)),
    {
      title: "Persistence",
    }
  );
}
var count = 0;

function uid(name) {
  return new Id("O-" + (name == null ? "" : name + "-") + ++count);
}

function Id(id) {
  this.id = id;
  this.href = new URL(`#${id}`, location) + "";
}

Id.prototype.toString = function () {
  return "url(" + this.href + ")";
};

function render_treemap(
  treep_map_root,
  width,
  height,
  color,
  data_name = "dataset_name"
) {
  var persistences = [];
  treep_map_root.each((d) => {
    if (d.data.persistence != "nan") persistences.push(d.data.persistence);
  });

  var min = d3.extent(persistences)[0];
  var max = d3.extent(persistences)[1];

  const color_scale = d3.scaleLinear(interpolatePoints(min, max, 7), color);

  const svg = d3
    .create("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", [0, 0, width, height])
    .attr(
      "style",
      "max-width: 100%; height: auto; overflow: visible; font: 10px sans-serif;"
    )
    .attr("id", "treemap")
    .attr("dataset_name", data_name);

  const shadow = uid("shadow");

  svg
    .append("filter")
    .attr("id", shadow.id)
    .append("feDropShadow")
    .attr("flood-opacity", 0.3)
    .attr("dx", 0)
    .attr("stdDeviation", 3);

  var grouped = d3.group(treep_map_root, (d) => d.depth);

  const node = svg
    .selectAll("g")
    .data(grouped)
    .join("g")
    .attr("filter", shadow)
    .selectAll("g")
    .data((d) => d[1])
    .join("g")
    .attr("transform", (d) => `translate(${d.x0},${d.y0})`);

  format = d3.format(",d");
  const format_float = d3.format(".2f");
  node.append("title").text(
    (d) =>
      `${d
        .ancestors()
        .reverse()
        .map((d) => d.data.name)
        .join("/")}\n value:${d.value} \n area:${format_float(
        (d.y1 - d.y0) * (d.x1 - d.x0)
      )} \n persistence:${format_float(d.data.persistence)}`
  );

  node
    .append("rect")
    .attr("id", (d) => (d.nodeUid = uid("node")).id)
    .attr("name", (d) => d.data.name)
    .attr("selected", (d) => d.data.flag)
    .attr("fill", (d) => {
      if (d.data.persistence != "nan") {
        return color_scale(d.data.persistence);
      } else return "gray";
    })
    .attr("width", (d) => d.x1 - d.x0)
    .attr("height", (d) => d.y1 - d.y0)
    .attr("style", (d) => {
      if (d.data.flag) {
        return "stroke-width:3;stroke:yellow";
      } else {
        return "none";
      }
    })
    .on("click", function (event) {
      handleClick(event, svg);
    });

  node
    .append("clipPath")
    .attr("id", (d) => (d.clipUid = uid("clip")).id)
    .append("use")
    .attr("xlink:href", (d) => d.nodeUid.href);

  node
    .append("text")
    .attr("clip-path", (d) => d.clipUid)
    .attr("fill", (d) => {
      const bgColor = color_scale(d.data.persistence); // string rgb(r,g,b)
      if (!bgColor) {
        return "white"; // color for the root
      }
      const rgb = bgColor.match(/\d+/g);
      const r = parseInt(rgb[0]);
      const g = parseInt(rgb[1]);
      const b = parseInt(rgb[2]);

      const brightness = (r * 299 + g * 587 + b * 114) / 1000;

      return brightness > 128 ? "black" : "white";
    })
    .selectAll("tspan")
    .data((d) => d.data.name.split(/(?=[A-Z][^A-Z])/g).concat(format(d.value)))
    .join("tspan")
    .attr("fill-opacity", (d, i, nodes) =>
      i === nodes.length - 1 ? 0.7 : null
    )
    .text((d) => d);

  node
    .filter((d) => d.children)
    .selectAll("tspan")
    .attr("dx", 3)
    .attr("y", 13);

  node
    .filter((d) => !d.children)
    .selectAll("tspan")
    .attr("x", 3)
    .attr(
      "y",
      (d, i, nodes) => `${(i === nodes.length - 1) * 0.3 + 1.1 + i * 0.9}em`
    );

  return svg.node();
}

function root_from_data(data, selected_components = []) {
  const root = d3.hierarchy(data);

  root.each((d) => (d.value = +d.data.value));
  root.each(
    (d) => (d.data.flag = selected_components.includes(parseInt(d.data.name)))
  );
  root.sort((a, b) => b.height - a.height || b.value - a.value);

  return root;
}

function get_all_flags(svg) {
  const flags = {};

  svg.selectAll("rect").each(function (d) {
    flags[d3.select(this).attr("name")] =
      d3.select(this).attr("selected") === "true";
  });

  return flags;
}

function get_all_persistences(root) {
  const persistences = {};
  root.each((d) => (persistences[d.data.name] = d.data.persistence));
  return persistences;
}

function treemap_from_root(
  data,
  tile_method,
  width,
  height,
  nested,
  paddingOuter = 3,
  paddingTop = 19,
  paddingIneer = 1
) {
  var tree = 0;
  if (nested == true) {
    tree = d3
      .treemap()
      .tile(tile_method)
      .size([width, height])
      .round(false)
      .paddingOuter(paddingOuter)
      .paddingTop(paddingTop)
      .paddingInner(paddingIneer)(data);
  } else {
    tree = d3.treemap().tile(tile_method).size([width, height]).round(false)(
      data
    );
  }

  return tree;
}

function render_from_data(
  data,
  tile_method,
  width,
  height,
  nested,
  color,
  data_name = "",
  selected_components = [],
  paddingOuter = 3,
  paddingTop = 19,
  paddingIneer = 1
) {
  var root = root_from_data(data, selected_components);
  var tree_map = treemap_from_root(
    root,
    tile_method,
    width,
    height,
    nested,
    paddingOuter,
    paddingTop,
    paddingIneer
  );
  return render_treemap(tree_map, width, height, color, data_name);
}

function update_flag_all_childs(root, name) {
  var node_target = root.find((d) => d.data.name == name);
  node_target.eachAfter((d) => (d.data.flag = !node_target.data.flag));
  return node_target;
}

function render_ploty_plot(k_components) {
  try {
    update_selecteds();
  } catch {}

  var cur_dataset = document.forms["dataset-form"].elements["dataset"].value;
  const eta_val = document.forms["dataset-form"].elements["eta"].value;
  const column_to_color = document.getElementById("column-select").value;
  fetch("/plotly-plot", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      global_selecteds: k_components,
      dataset_name: cur_dataset,
      eta: eta_val,
      column_to_color: column_to_color,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      const x = JSON.parse(data);

      const plotlyDiv = document.getElementById("plotly-div");
      Plotly.react(plotlyDiv, x.data, x.layout);
    });
}

function appendSvg(svg) {
  return new Promise((resolve, reject) => {
    document.getElementById("d3-svg").appendChild(svg);
    resolve();
  });
}

async function fetch_leaves(dataset_name, eta_val) {
  try {
    const response = await fetch("/get_leaves", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: dataset_name,
        eta: eta_val,
      }),
    });
    const data = await response.json();
    const x = data.components;

    return x;
  } catch (error) {
    console.error("Erro:", error);
  }
}

function update_selecteds() {
  svgElement = document.getElementById("treemap");

  const rects = svgElement.querySelectorAll('rect[selected="true"]');
  const selectedNames = [];
  rects.forEach((rect) => {
    selectedNames.push(rect.getAttribute("name"));
  });
  global_selecteds = selectedNames;
}

function fetchDatasets() {
  fetch("/datasets")
    .then((response) => response.json())
    .then((data) => {
      const datasetSelect = document.getElementById("dataset");
      datasetSelect.innerHTML = "";
      data.datasets.forEach((dataset) => {
        const option = document.createElement("option");
        option.value = dataset;
        option.text = dataset;
        datasetSelect.appendChild(option);
      });
    })
    .then((d) => {
      fetchColumns();
    });
}

function fetchColumns() {
  const datasetSelect = document.getElementById("dataset");
  const selectedDataset = datasetSelect.value;

  fetch(`/columns?dataset=${selectedDataset}`)
    .then((response) => response.json())
    .then((data) => {
      const columnSelect = document.getElementById("column-select");
      columnSelect.innerHTML = "";
      data.columns.forEach((column) => {
        const option = document.createElement("option");
        option.value = column;
        option.text = column;
        columnSelect.appendChild(option);
      });
    })
    .then((d) => {
      var cur_dataset =
        document.forms["dataset-form"].elements["dataset"].value;
      const eta_val = document.forms["dataset-form"].elements["eta"].value;
      const column_to_color = document.getElementById("column-select").value;
      fetch("/plotly-plot", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          global_selecteds: global_selecteds,
          dataset_name: cur_dataset,
          eta: eta_val,
          column_to_color: column_to_color,
        }),
      })
        .then((response) => response.json())
        .then((data) => {
          const x = JSON.parse(data);

          const plotlyDiv = document.getElementById("plotly-div");
          Plotly.react(plotlyDiv, x.data, x.layout);
        });
    });
}

function plotlySubmit() {
  try {
    update_selecteds();
  } catch {}

  var cur_dataset = document.forms["dataset-form"].elements["dataset"].value;
  const eta_val = document.forms["dataset-form"].elements["eta"].value;
  const column_to_color = document.getElementById("column-select").value;
  fetch("/plotly-plot", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      global_selecteds: global_selecteds,
      dataset_name: cur_dataset,
      eta: eta_val,
      column_to_color: column_to_color,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      const x = JSON.parse(data);

      const plotlyDiv = document.getElementById("plotly-div");
      Plotly.react(plotlyDiv, x.data, x.layout);
    });
}

// Listeners
document.addEventListener("DOMContentLoaded", function () {
  fetchDatasets();

  document
    .getElementById("dataset-form")
    .addEventListener("submit", function (e) {
      e.preventDefault();
      const dataset_name =
        document.forms["dataset-form"].elements["dataset"].value;
      const eta_val = document.forms["dataset-form"].elements["eta"].value;

      fetch_leaves(dataset_name, eta_val).then((components) => {
        fetch("/get-dataset", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ dataset_name: dataset_name, eta: eta_val }),
        })
          .then((response) => {
            if (!response.ok) {
              throw new Error(
                "Network response was not ok " + response.statusText
              );
            }
            return response.json();
          })
          .then((data) => {
            var container = document.querySelector(".svg-container");
            var width = container.getBoundingClientRect().width;
            var height = container.getBoundingClientRect().height - 55;
            const my_svg = render_from_data(
              data,
              tile,
              width,
              height,
              true,
              color,
              (data_name = dataset_name),
              (selected_componets = components)
            );
            const legend = make_legend(data, color, width);
            element = document.getElementById("d3-svg");
            while (element.firstChild) {
              element.removeChild(element.firstChild);
            }
            document.getElementById("d3-svg").appendChild(legend);

            appendSvg(my_svg).then(render_ploty_plot(components));
          })
          .catch((error) => {
            console.error(
              "There has been a problem with your fetch operation:",
              error
            );
          });
      });
    });

  const colorColumnSelect = document.getElementById("column-select");
  colorColumnSelect.addEventListener("change", function () {
    plotlySubmit();
  });
});

window.addEventListener("resize", function () {
  const plotlyDiv = document.getElementById("plotly-div");
  Plotly.relayout(plotlyDiv, {
    "xaxis.autorange": true,
    "yaxis.autorange": true,
  });
});

document.addEventListener("keydown", function (event) {
  if (event.ctrlKey) {
    isCtrlPressed = true;
  } else if (event.key === "c" || event.key === "C") {
    let svg = d3.select(document.getElementById("treemap"));

    clearSelection(svg);

    const flags = get_all_flags(svg);

    updateSelections(svg, flags);
    highlightSelections(svg, flags);
    update_selecteds();
  }
});

document.addEventListener("keyup", function (event) {
  if (event.key === "Control") {
    isCtrlPressed = false;
    plotlySubmit();
  }
});
