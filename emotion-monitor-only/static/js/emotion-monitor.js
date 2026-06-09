'use strict';

/* =========================================================
 * API 配置
 * ========================================================= */
var EMOTION_API = {
  modelStatus: '/emotion-monitor/api/model-status/',
  dashboardData: '/emotion-monitor/api/dashboard-data/',
  predictTrialSeparated: '/emotion-monitor/api/predict-trial-separated/'
};

/* =========================================================
 * 全局状态
 * ========================================================= */
window.currentEmotionState = {
  valence: 0,
  arousal: 0,
  confidence: 0,
  color: '#7367f0',
  emotion_label: 'Waiting',
  emotion_type: '等待识别'
};

window.emotionTrendState = {
  categories: [],
  valence: [],
  arousal: []
};

window.rawTopomapState = {
  bands: [],
  currentBandIndex: 0
};

window.eegCharts = {};
window.vaChart = null;
window.peripheralChart = null;
window.emotionTrendChart = null;

window.rawEEGPreviewData = null;
window.rawPeripheralPreviewData = null;

/* =========================================================
 * 页面初始化
 * ========================================================= */
document.addEventListener('DOMContentLoaded', function () {
  console.log('[Emotion Monitor] JS 已加载');

  updateCurrentTime();
  setInterval(updateCurrentTime, 1000);

  initAllCharts();

  checkModelStatus();

  bindTrialSeparatedUpload();
  bindTrialVideoPreview();

  bindThemeObserver();
});

/* =========================================================
 * 主题变化监听
 * ========================================================= */
function bindThemeObserver() {
  var observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (m) {
      if (m.attributeName === 'class') {
        console.log('[Emotion Monitor] 主题变化，重绘图表');

        destroyAllCharts();
        initAllCharts();

        if (window.currentEmotionState) {
          updateVAChart(
            window.currentEmotionState.valence,
            window.currentEmotionState.arousal,
            window.currentEmotionState.color
          );
        }

        refreshEmotionTrendChart();
        renderCurrentRawTopomap();
        applyRawEEGPreviewData(window.rawEEGPreviewData);
        applyRawPeripheralPreviewData(window.rawPeripheralPreviewData);
      }
    });
  });

  observer.observe(document.documentElement, {
    attributes: true
  });
}

/* =========================================================
 * CSRF 工具
 * ========================================================= */
function getCookie(name) {
  var cookieValue = null;

  if (document.cookie && document.cookie !== '') {
    var cookies = document.cookie.split(';');

    for (var i = 0; i < cookies.length; i++) {
      var cookie = cookies[i].trim();

      if (cookie.substring(0, name.length + 1) === name + '=') {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }

  return cookieValue;
}

function getCsrfToken() {
  var input = document.querySelector('input[name="csrfmiddlewaretoken"]');

  if (input && input.value) {
    return input.value;
  }

  return getCookie('csrftoken');
}

/* =========================================================
 * 主题判断
 * ========================================================= */
function isDark() {
  return document.documentElement.classList.contains('dark-style');
}

function getThemeMode() {
  return isDark() ? 'dark' : 'light';
}

function getLabelColor() {
  return isDark() ? '#cbd5e1' : '#475569';
}

function getGridColor() {
  return isDark() ? '#334155' : '#e2e8f0';
}

function getAxisColor() {
  return isDark() ? '#64748b' : '#94a3b8';
}

/* =========================================================
 * 时间
 * ========================================================= */
function updateCurrentTime() {
  var el = document.getElementById('currentTime');

  if (!el) {
    return;
  }

  el.innerText = new Date().toLocaleTimeString();
}

/* =========================================================
 * 图表初始化 / 销毁
 * ========================================================= */
function safeRun(name, fn) {
  try {
    if (typeof fn === 'function') {
      fn();
    } else {
      console.warn('[Emotion Monitor] 函数未定义：', name);
    }
  } catch (error) {
    console.error('[Emotion Monitor] 初始化失败：' + name, error);
  }
}

function initAllCharts() {
  safeRun('renderVAChart', renderVAChart);
  safeRun('renderEEGChart', renderEEGChart);
  safeRun('renderPeripheralChart', renderPeripheralChart);
  safeRun('initRawTopomapView', initRawTopomapView);
  safeRun('renderEmotionTrendChart', renderEmotionTrendChart);

  setTimeout(function () {
    if (window.rawEEGPreviewData || window.rawPeripheralPreviewData) {
      applyRawEEGPreviewData(window.rawEEGPreviewData);
      applyRawPeripheralPreviewData(window.rawPeripheralPreviewData);
    } else {
      updateSignalChartsByTrial(1);
    }
  }, 300);
}

function destroyAllCharts() {
  if (window.vaChart) {
    window.vaChart.destroy();
    window.vaChart = null;
  }

  if (window.eegCharts) {
    Object.keys(window.eegCharts).forEach(function (key) {
      if (window.eegCharts[key]) {
        window.eegCharts[key].destroy();
      }
    });

    window.eegCharts = {};
  }

  if (window.peripheralChart) {
    window.peripheralChart.destroy();
    window.peripheralChart = null;
  }

  if (window.emotionTrendChart) {
    window.emotionTrendChart.destroy();
    window.emotionTrendChart = null;
  }
}

/* =========================================================
 * VA 散点图
 * ========================================================= */
function renderVAChart() {
  var el = document.querySelector('#vaChart');

  if (!el) {
    console.warn('[Emotion Monitor] #vaChart 未找到，跳过 VA 图初始化');
    return;
  }

  if (typeof ApexCharts === 'undefined') {
    console.error('[Emotion Monitor] ApexCharts 未加载');
    return;
  }

  var options = {
    chart: {
      type: 'scatter',
      height: 240,
      toolbar: {
        show: false
      },
      animations: {
        enabled: true
      },
      background: 'transparent'
    },
    theme: {
      mode: getThemeMode()
    },
    series: [
      {
        name: 'Current Emotion',
        data: [[0, 0]]
      }
    ],
    colors: ['#7367f0'],
    markers: {
      size: 12,
      colors: ['#7367f0'],
      strokeColors: isDark() ? '#ffffff' : '#1e293b',
      strokeWidth: 2,
      hover: {
        size: 15
      }
    },
    xaxis: {
      min: -1,
      max: 1,
      tickAmount: 4,
      title: {
        text: 'Valence',
        style: {
          color: getLabelColor(),
          fontSize: '11px'
        }
      },
      labels: {
        style: {
          colors: getLabelColor()
        },
        formatter: function (v) {
          return Number(v).toFixed(1);
        }
      },
      axisBorder: {
        color: getAxisColor()
      },
      axisTicks: {
        color: getAxisColor()
      }
    },
    yaxis: {
      min: -1,
      max: 1,
      tickAmount: 4,
      title: {
        text: 'Arousal',
        style: {
          color: getLabelColor(),
          fontSize: '11px'
        }
      },
      labels: {
        style: {
          colors: getLabelColor()
        },
        formatter: function (v) {
          return Number(v).toFixed(1);
        }
      }
    },
    grid: {
      borderColor: getGridColor(),
      strokeDashArray: 4,
      xaxis: {
        lines: {
          show: true
        }
      },
      yaxis: {
        lines: {
          show: true
        }
      }
    },
    legend: {
      show: false
    },
    tooltip: {
      theme: getThemeMode(),
      x: {
        formatter: function (v) {
          return 'Valence: ' + Number(v).toFixed(2);
        }
      },
      y: {
        formatter: function (v) {
          return 'Arousal: ' + Number(v).toFixed(2);
        }
      }
    },
    annotations: {
      xaxis: [
        {
          x: 0,
          borderColor: getAxisColor(),
          strokeDashArray: 5
        }
      ],
      yaxis: [
        {
          y: 0,
          borderColor: getAxisColor(),
          strokeDashArray: 5
        }
      ],
      points: [
        {
          x: 0.55,
          y: 0.75,
          marker: {
            size: 0
          },
          label: {
            text: 'Excited',
            borderColor: 'transparent',
            style: {
              color: '#ff9f43',
              background: 'transparent',
              fontSize: '11px'
            }
          }
        },
        {
          x: 0.55,
          y: -0.55,
          marker: {
            size: 0
          },
          label: {
            text: 'Calm',
            borderColor: 'transparent',
            style: {
              color: '#28c76f',
              background: 'transparent',
              fontSize: '11px'
            }
          }
        },
        {
          x: -0.75,
          y: 0.75,
          marker: {
            size: 0
          },
          label: {
            text: 'Tense',
            borderColor: 'transparent',
            style: {
              color: '#ea5455',
              background: 'transparent',
              fontSize: '11px'
            }
          }
        },
        {
          x: -0.65,
          y: -0.55,
          marker: {
            size: 0
          },
          label: {
            text: 'Sad',
            borderColor: 'transparent',
            style: {
              color: '#7367f0',
              background: 'transparent',
              fontSize: '11px'
            }
          }
        }
      ]
    }
  };

  window.vaChart = new ApexCharts(el, options);
  window.vaChart.render();

  console.log('[Emotion Monitor] VA Chart 初始化完成');
}

function updateVAChart(valence, arousal, color) {
  valence = Number(valence || 0);
  arousal = Number(arousal || 0);

  var pointColor = color || getVAColor(valence, arousal);

  if (!window.vaChart) {
    console.warn('[Emotion Monitor] VA Chart 尚未初始化');
    return;
  }

  window.vaChart.updateOptions({
    markers: {
      size: 12,
      colors: [pointColor],
      strokeColors: isDark() ? '#ffffff' : '#1e293b',
      strokeWidth: 2,
      hover: {
        size: 15
      }
    }
  });

  window.vaChart.updateSeries([
    {
      name: 'Current Emotion',
      data: [[Number(valence.toFixed(2)), Number(arousal.toFixed(2))]]
    }
  ]);
}

function getVAColor(valence, arousal) {
  valence = Number(valence || 0);
  arousal = Number(arousal || 0);

  if (valence >= 0 && arousal >= 0) {
    return '#ffb020';
  }

  if (valence >= 0 && arousal < 0) {
    return '#28c76f';
  }

  if (valence < 0 && arousal >= 0) {
    return '#ea5455';
  }

  return '#7367f0';
}

/* =========================================================
 * 原始 EEG 四行独立线图
 * ========================================================= */
function renderEEGChart() {
  window.eegCharts = {};

  if (typeof ApexCharts === 'undefined') {
    console.error('[Emotion Monitor] ApexCharts 未加载');
    return;
  }

  var channelConfigs = [
    {
      key: 'Fp1',
      id: 'eegChartFp1',
      color: '#00cfe8'
    },
    {
      key: 'F3',
      id: 'eegChartF3',
      color: '#7367f0'
    },
    {
      key: 'F4',
      id: 'eegChartF4',
      color: '#28c76f'
    },
    {
      key: 'Cz',
      id: 'eegChartCz',
      color: '#ff9f43'
    }
  ];

  channelConfigs.forEach(function (cfg) {
    var el = document.getElementById(cfg.id);

    if (!el) {
      console.warn('[Emotion Monitor] EEG 容器未找到：', cfg.id);
      return;
    }

    var seed = 17;
    var amp = 0.7;
    if (cfg.key === 'Fp1') { amp = 0.8; seed += 1; }
    else if (cfg.key === 'F3') { amp = 0.6; seed += 2; }
    else if (cfg.key === 'F4') { amp = 0.5; seed += 3; }
    else if (cfg.key === 'Cz') { amp = 0.7; seed += 4; }

    var initData = generateWaveDataWithSeed(300, amp, 0.3, seed);
    var options = createSingleEEGChartOptions(cfg.key, cfg.color, initData);

    window.eegCharts[cfg.key] = new ApexCharts(el, options);
    window.eegCharts[cfg.key].render();
  });

  console.log('[Emotion Monitor] EEG 四通道图初始化完成');
}

function createSingleEEGChartOptions(channelName, color, data) {
  var categories = [];

  for (var i = 0; i < 300; i++) {
    categories.push(i);
  }

  return {
    chart: {
      type: 'line',
      height: 78,
      toolbar: {
        show: false
      },
      animations: {
        enabled: true
      },
      sparkline: {
        enabled: false
      },
      background: 'transparent'
    },
    theme: {
      mode: getThemeMode()
    },
    series: [
      {
        name: channelName,
        data: data || createZeroData(300)
      }
    ],
    colors: [color],
    stroke: {
      curve: 'smooth',
      width: 1.8
    },
    xaxis: {
      categories: categories,
      labels: {
        show: false
      },
      axisBorder: {
        show: false
      },
      axisTicks: {
        show: false
      },
      tooltip: {
        enabled: false
      }
    },
    yaxis: {
      labels: {
        show: true,
        minWidth: 28,
        style: {
          colors: getLabelColor(),
          fontSize: '10px'
        },
        formatter: function (v) {
          return Number(v).toFixed(1);
        }
      }
    },
    grid: {
      borderColor: getGridColor(),
      strokeDashArray: 3,
      padding: {
        left: 0,
        right: 8,
        top: 0,
        bottom: 0
      }
    },
    legend: {
      show: false
    },
    tooltip: {
      theme: getThemeMode(),
      x: {
        show: false
      },
      y: {
        formatter: function (v) {
          return channelName + ': ' + Number(v).toFixed(3);
        }
      }
    }
  };
}

/* =========================================================
 * 原始外周信号图
 * ========================================================= */
function renderPeripheralChart() {
  var categories = [];

  for (var i = 0; i < 300; i++) {
    categories.push(i);
  }

  var options = {
    chart: {
      type: 'line',
      height: 280,
      toolbar: {
        show: false
      },
      background: 'transparent'
    },
    theme: {
      mode: getThemeMode()
    },
    series: [
      {
        name: 'GSR',
        data: generateSlowDataWithSeed(300, 0.4, 32)
      },
      {
        name: 'Resp',
        data: generateWaveDataWithSeed(300, 0.8, 0.2, 33)
      },
      {
        name: 'Pleth',
        data: generateWaveDataWithSeed(300, 0.9, 0.2, 34)
      }
    ],
    colors: ['#28c76f', '#ff9f43', '#7367f0'],
    stroke: {
      curve: 'smooth',
      width: 2
    },
    xaxis: {
      categories: categories,
      labels: {
        show: false
      },
      axisBorder: {
        color: getGridColor()
      },
      axisTicks: {
        show: false
      }
    },
    yaxis: {
      labels: {
        style: {
          colors: getLabelColor()
        }
      },
      title: {
        text: 'Normalized Raw Peripheral',
        style: {
          color: getLabelColor()
        }
      }
    },
    grid: {
      borderColor: getGridColor(),
      strokeDashArray: 4
    },
    legend: {
      labels: {
        colors: getLabelColor()
      }
    },
    tooltip: {
      theme: getThemeMode()
    },
    noData: {
      text: '等待上传原始外周数据'
    }
  };

  var el = document.querySelector('#peripheralChart');

  if (el) {
    window.peripheralChart = new ApexCharts(el, options);
    window.peripheralChart.render();
    console.log('[Emotion Monitor] Peripheral Chart 初始化完成');
  } else {
    console.warn('[Emotion Monitor] #peripheralChart 未找到');
  }
}

/* =========================================================
 * 情绪趋势图
 * ========================================================= */
function renderEmotionTrendChart() {
  var options = {
    chart: {
      type: 'line',
      height: 280,
      toolbar: {
        show: false
      },
      background: 'transparent'
    },
    theme: {
      mode: getThemeMode()
    },
    series: [
      {
        name: 'Valence',
        data: window.emotionTrendState.valence || []
      },
      {
        name: 'Arousal',
        data: window.emotionTrendState.arousal || []
      }
    ],
    colors: ['#28c76f', '#ff9f43'],
    stroke: {
      curve: 'smooth',
      width: 3
    },
    xaxis: {
      categories: window.emotionTrendState.categories || [],
      labels: {
        style: {
          colors: getLabelColor()
        }
      }
    },
    yaxis: {
      min: -1,
      max: 1,
      labels: {
        style: {
          colors: getLabelColor()
        }
      }
    },
    grid: {
      borderColor: getGridColor(),
      strokeDashArray: 4
    },
    legend: {
      labels: {
        colors: getLabelColor()
      }
    },
    tooltip: {
      theme: getThemeMode()
    }
  };

  var el = document.querySelector('#emotionTrendChart');

  if (el) {
    window.emotionTrendChart = new ApexCharts(el, options);
    window.emotionTrendChart.render();
    console.log('[Emotion Monitor] Emotion Trend Chart 初始化完成');
  } else {
    console.warn('[Emotion Monitor] #emotionTrendChart 未找到');
  }
}

/* =========================================================
 * 原始 EEG 脑地形图
 * ========================================================= */
function initRawTopomapView() {
  ensureRawTopomapDom();
  bindRawTopomapBandSelect();
  renderCurrentRawTopomap();
}

function ensureRawTopomapDom() {
  var img = document.getElementById('rawTopomapImage');
  var container = document.getElementById('eegTopomapChart');

  if (img) {
    return;
  }

  if (container) {
    container.innerHTML =
      '<div class="text-center">' +
      '<img id="rawTopomapImage" alt="Raw EEG Topomap" ' +
      'style="max-width: 100%; height: 260px; object-fit: contain; display: none;" />' +
      '<div id="rawTopomapPlaceholder" class="text-muted small py-5">等待上传原始 EEG 数据生成脑地形图</div>' +
      '</div>';
  }
}

function getRawTopomapSelect() {
  return document.getElementById('rawTopomapBandSelect') || document.getElementById('topomapBandSelect');
}

function bindRawTopomapBandSelect() {
  var select = getRawTopomapSelect();

  if (!select) {
    return;
  }

  if (select.dataset.bound === '1') {
    return;
  }

  select.addEventListener('change', function () {
    window.rawTopomapState.currentBandIndex = Number(select.value || 0);
    renderCurrentRawTopomap();
  });

  select.dataset.bound = '1';
}

function updateRawTopomap(rawTopomap) {
  if (!rawTopomap || !rawTopomap.bands || !rawTopomap.bands.length) {
    console.warn('[Emotion Monitor] raw_topomap 为空');
    return;
  }

  window.rawTopomapState = {
    bands: rawTopomap.bands,
    currentBandIndex: 0
  };

  updateRawTopomapBandSelect(rawTopomap.bands);
  renderCurrentRawTopomap();
}

function updateRawTopomapBandSelect(bands) {
  var select = getRawTopomapSelect();

  if (!select) {
    return;
  }

  select.innerHTML = '';

  for (var i = 0; i < bands.length; i++) {
    var option = document.createElement('option');
    option.value = String(i);
    option.innerText = bands[i].name || 'Band ' + (i + 1);
    select.appendChild(option);
  }

  select.value = '0';
  bindRawTopomapBandSelect();
}

function renderCurrentRawTopomap() {
  var img = document.getElementById('rawTopomapImage');
  var placeholder = document.getElementById('rawTopomapPlaceholder');

  if (!img || !window.rawTopomapState) {
    return;
  }

  var bands = window.rawTopomapState.bands || [];
  var index = window.rawTopomapState.currentBandIndex || 0;

  if (!bands.length) {
    img.removeAttribute('src');
    img.style.display = 'none';

    if (placeholder) {
      placeholder.style.display = 'block';
    }

    return;
  }

  if (index < 0 || index >= bands.length) {
    index = 0;
  }

  img.src = bands[index].image || '';
  img.style.display = bands[index].image ? 'inline-block' : 'none';

  if (placeholder) {
    placeholder.style.display = bands[index].image ? 'none' : 'block';
  }
}

/* =========================================================
 * 模型状态检查
 * ========================================================= */
function checkModelStatus() {
  fetch(EMOTION_API.modelStatus, {
    credentials: 'same-origin'
  })
    .then(function (res) {
      return res.json();
    })
    .then(function (data) {
      console.log('[Emotion Monitor] 模型状态：', data);

      var modelVersion = document.getElementById('modelVersion');

      if (modelVersion) {
        if (data.model_loaded) {
          modelVersion.innerText = 'MH-DGFNet Ready';
          modelVersion.className = 'badge bg-label-success';
        } else {
          modelVersion.innerText = 'Model Error';
          modelVersion.className = 'badge bg-label-danger';
        }
      }

      var modelStatusText = document.getElementById('modelStatusText');

      if (modelStatusText) {
        modelStatusText.innerText = data.model_loaded ? '模型已加载' : data.model_error || '模型异常';
      }
    })
    .catch(function (err) {
      console.error('[Emotion Monitor] 模型状态检查失败：', err);

      var modelVersion = document.getElementById('modelVersion');

      if (modelVersion) {
        modelVersion.innerText = 'Model Status Error';
        modelVersion.className = 'badge bg-label-danger';
      }
    });
}

/* =========================================================
 * 三模态上传识别
 * ========================================================= */
function bindTrialSeparatedUpload() {
  var button = document.getElementById('predictTrialBtn');
  var statusText = document.getElementById('predictStatusText');

  if (!button) {
    console.warn('[Emotion Monitor] predictTrialBtn 未找到');
    return;
  }

  if (button.dataset.bound === '1') {
    return;
  }

  button.addEventListener('click', function () {
    var featureInput = document.getElementById('featureNpzInput');
    var visualInput = document.getElementById('visualNpyInput');
    var rawDataInput = document.getElementById('rawDataInput');
    var trialInput = document.getElementById('trialIndexInput');

    console.log('[Emotion Monitor] 上传控件检查：', {
      featureInput: featureInput,
      visualInput: visualInput,
      rawDataInput: rawDataInput,
      trialInput: trialInput
    });

    if (!featureInput || !visualInput || !rawDataInput || !trialInput) {
      alert('上传控件未找到，请检查 HTML 中 featureNpzInput / visualNpyInput / rawDataInput / trialIndexInput 是否存在');
      return;
    }

    var featureFile = featureInput.files && featureInput.files[0] ? featureInput.files[0] : null;
    var visualFile = visualInput.files && visualInput.files[0] ? visualInput.files[0] : null;
    var rawFile = rawDataInput.files && rawDataInput.files[0] ? rawDataInput.files[0] : null;
    var trialIndex = trialInput.value || '1';

    console.log('[Emotion Monitor] 选择的文件：', {
      featureFile: featureFile ? featureFile.name : null,
      visualFile: visualFile ? visualFile.name : null,
      rawFile: rawFile ? rawFile.name : null,
      rawFileSize: rawFile ? rawFile.size : null,
      trialIndex: trialIndex
    });

    if (!featureFile) {
      alert('请先选择 EEG/外周 NPZ 特征文件，例如 s03.npz');
      return;
    }

    if (!visualFile) {
      alert('请先选择当前 Trial 的视觉特征 NPY 文件，例如 s03_trial01_features.npy');
      return;
    }

    if (!rawFile) {
      alert('请先选择原始 EEG/外周数据文件，例如 DEAP 原始 s03.dat。原始波形和脑地形图必须基于原始数据。');
      return;
    }

    var rawFileName = rawFile.name || '';

    if (!rawFileName.toLowerCase().endsWith('.dat')) {
      alert('当前后端只支持 DEAP 原始 .dat 文件，请上传例如 s03.dat 的原始数据文件。');
      return;
    }

    if (!trialIndex) {
      alert('请输入 Trial 编号，范围 1~40');
      return;
    }

    var trialNumber = Number(trialIndex);

    if (isNaN(trialNumber) || trialNumber < 1 || trialNumber > 40) {
      alert('Trial 编号必须是 1~40');
      return;
    }

    var formData = new FormData();

    formData.append('feature_npz', featureFile);
    formData.append('visual_npy', visualFile);
    formData.append('raw_data_file', rawFile);
    formData.append('trial_index', trialIndex);

    var csrfToken = getCsrfToken();

    console.log('[Emotion Monitor] CSRF Token：', csrfToken ? '已获取' : '未获取');

    console.log('[Emotion Monitor] 即将提交 FormData：');

    formData.forEach(function (value, key) {
      if (value instanceof File) {
        console.log(key, value.name, value.size, value.type);
      } else {
        console.log(key, value);
      }
    });

    button.disabled = true;
    button.innerText = '识别中...';

    if (statusText) {
      statusText.innerText = '正在上传原始数据与特征文件，并调用 MH-DGFNet 推理，请稍候...';
    }

    fetch(EMOTION_API.predictTrialSeparated, {
      method: 'POST',
      body: formData,
      credentials: 'same-origin',
      headers: {
        'X-CSRFToken': csrfToken || ''
      }
    })
      .then(function (res) {
        return res.text().then(function (text) {
          var data = null;

          try {
            data = JSON.parse(text);
          } catch (e) {
            console.error('[Emotion Monitor] 接口返回的不是 JSON');
            console.error('[Emotion Monitor] HTTP 状态码：', res.status);
            console.error('[Emotion Monitor] 原始返回内容：', text);

            throw new Error(
              '接口返回非 JSON，HTTP 状态码：' + res.status + '。通常是 CSRF 403、URL 错误或后端异常 HTML 页面。'
            );
          }

          if (!res.ok) {
            console.error('[Emotion Monitor] 接口 HTTP 错误：', res.status, data);
            throw data;
          }

          return data;
        });
      })
      .then(function (data) {
        console.log('[Emotion Monitor] 三模态识别结果：', data);

        if (!data.ok) {
          alert(data.error || '三模态识别失败');

          if (statusText) {
            statusText.innerText = '识别失败：' + (data.error || '未知错误');
          }

          return;
        }

        applyPredictionDashboard(data.dashboard, data.trial_index);
        closeTrialUploadModal();

        if (statusText) {
          var emotion = data.dashboard && data.dashboard.current_emotion ? data.dashboard.current_emotion : {};

          statusText.innerText =
            '识别完成：Trial ' +
            data.trial_index +
            '，情绪=' +
            (emotion.emotion_label || '--') +
            '，Valence=' +
            formatNumber(emotion.valence) +
            '，Arousal=' +
            formatNumber(emotion.arousal) +
            '，Confidence=' +
            formatPercent(emotion.confidence);
        }
      })
      .catch(function (err) {
        console.error('[Emotion Monitor] 三模态识别接口请求失败：', err);

        var message = '三模态识别接口请求失败，请查看控制台';

        if (err && err.error) {
          message = err.error;
        } else if (err && err.message) {
          message = err.message;
        }

        alert(message);

        if (statusText) {
          statusText.innerText = '接口请求失败：' + message;
        }
      })
      .finally(function () {
        button.disabled = false;
        button.innerText = '开始识别';
      });
  });

  button.dataset.bound = '1';
}

function closeTrialUploadModal() {
  var modalEl = document.getElementById('trialUploadModal');

  if (!modalEl || !window.bootstrap) {
    return;
  }

  var modal = bootstrap.Modal.getInstance(modalEl);

  if (!modal) {
    modal = new bootstrap.Modal(modalEl);
  }

  modal.hide();
}

/* =========================================================
 * 应用后端 dashboard 结果
 * ========================================================= */
function applyPredictionDashboard(dashboard, trialIndex) {
  if (!dashboard) {
    return;
  }

  if (dashboard.current_emotion) {
    var emotion = dashboard.current_emotion;

    window.currentEmotionState = {
      valence: Number(emotion.valence || 0),
      arousal: Number(emotion.arousal || 0),
      confidence: Number(emotion.confidence || 0),
      color: emotion.color || getVAColor(Number(emotion.valence || 0), Number(emotion.arousal || 0)),
      emotion_label: emotion.emotion_label || 'Unknown',
      emotion_type: emotion.emotion_type || '--'
    };

    updateEmotionCardFromBackend(emotion);
    updateVAChart(emotion.valence, emotion.arousal, emotion.color);
    updateEmotionTrendFromPrediction(emotion.valence, emotion.arousal);
  }

  if (dashboard.face) {
    updateFaceInfo(dashboard.face);
  }

  if (dashboard.peripheral) {
    updatePeripheralInfo(dashboard.peripheral);
  }

  if (dashboard.raw_signals) {
    updateRawSignalCharts(dashboard.raw_signals);
  } else if (dashboard.signals) {
    updateSignalChartsFromBackend(dashboard.signals);
  } else {
    updateSignalChartsByTrial(trialIndex);
  }

  if (dashboard.raw_topomap) {
    updateRawTopomap(dashboard.raw_topomap);
  }

  if (dashboard.model && dashboard.model.version) {
    setText('modelVersion', dashboard.model.version);
  }
}

/* =========================================================
 * 原始信号图更新
 * ========================================================= */
function updateRawSignalCharts(rawSignals) {
  if (!rawSignals) {
    return;
  }

  if (rawSignals.eeg_raw_preview) {
    window.rawEEGPreviewData = rawSignals.eeg_raw_preview;
  }

  if (rawSignals.peripheral_raw_preview) {
    window.rawPeripheralPreviewData = rawSignals.peripheral_raw_preview;
  }

  applyRawEEGPreviewData(window.rawEEGPreviewData);
  applyRawPeripheralPreviewData(window.rawPeripheralPreviewData);

  if (rawSignals.fs) {
    console.log('[Emotion Monitor] 原始数据采样率：', rawSignals.fs);
  }
}

function applyRawEEGPreviewData(eegData) {
  if (!eegData || !window.eegCharts) {
    return;
  }

  eegData.forEach(function (item) {
    if (!item || !item.name) {
      return;
    }

    var chart = window.eegCharts[item.name];

    if (chart) {
      chart.updateSeries(
        [
          {
            name: item.name,
            data: item.data || []
          }
        ]
      );
    } else {
      console.warn('[Emotion Monitor] 未找到 EEG 图表：', item.name, Object.keys(window.eegCharts));
    }
  });
}

function applyRawPeripheralPreviewData(periData) {
  if (!periData || !window.peripheralChart) {
    return;
  }

  window.peripheralChart.updateSeries(periData);
}

/* =========================================================
 * 处理后特征图更新，备用
 * ========================================================= */
function updateSignalChartsFromBackend(signals) {
  if (!signals) {
    return;
  }

  if (signals.eeg_preview && window.eegCharts) {
    signals.eeg_preview.forEach(function (item) {
      if (!item || !item.name) {
        return;
      }

      var chart = window.eegCharts[item.name];

      if (chart) {
        chart.updateSeries([
          {
            name: item.name,
            data: item.data || []
          }
        ]);
      }
    });
  }

  if (signals.peripheral_preview && window.peripheralChart) {
    window.peripheralChart.updateSeries(signals.peripheral_preview);
  }
}

/* =========================================================
 * 当前情绪卡片更新
 * ========================================================= */
function updateEmotionCardFromBackend(emotion) {
  if (!emotion) {
    return;
  }

  var emotionLabel = document.getElementById('emotionLabel');
  var emotionType = document.getElementById('emotionType');
  var valenceText = document.getElementById('valenceText');
  var arousalText = document.getElementById('arousalText');
  var confidenceText = document.getElementById('confidenceText');

  var valenceBar = document.getElementById('valenceBar');
  var arousalBar = document.getElementById('arousalBar');
  var confidenceBar = document.getElementById('confidenceBar');

  var valence = Number(emotion.valence || 0);
  var arousal = Number(emotion.arousal || 0);
  var confidence = Number(emotion.confidence || 0);

  var color = emotion.color || getVAColor(valence, arousal);

  if (emotionLabel) {
    emotionLabel.innerText = emotion.emotion_label || 'Unknown';
    emotionLabel.style.color = color;
  }

  if (emotionType) {
    emotionType.innerText = emotion.emotion_type || '--';
  }

  if (valenceText) {
    valenceText.innerText = valence.toFixed(2);
  }

  if (arousalText) {
    arousalText.innerText = arousal.toFixed(2);
  }

  if (confidenceText) {
    confidenceText.innerText = Math.round(confidence * 100) + '%';
  }

  if (valenceBar) {
    valenceBar.style.width = clampPercent(((valence + 1) / 2) * 100) + '%';
    valenceBar.style.backgroundColor = color;
  }

  if (arousalBar) {
    arousalBar.style.width = clampPercent(((arousal + 1) / 2) * 100) + '%';
    arousalBar.style.backgroundColor = color;
  }

  if (confidenceBar) {
    confidenceBar.style.width = clampPercent(confidence * 100) + '%';
    confidenceBar.style.backgroundColor = color;
  }
}

/* =========================================================
 * 情绪趋势更新
 * ========================================================= */
function updateEmotionTrendFromPrediction(valence, arousal) {
  valence = Number(valence || 0);
  arousal = Number(arousal || 0);

  if (!window.emotionTrendState) {
    window.emotionTrendState = {
      categories: [],
      valence: [],
      arousal: []
    };
  }

  var index = window.emotionTrendState.categories.length + 1;
  var label = 'Trial ' + index;

  window.emotionTrendState.categories.push(label);
  window.emotionTrendState.valence.push(Number(valence.toFixed(2)));
  window.emotionTrendState.arousal.push(Number(arousal.toFixed(2)));

  if (window.emotionTrendState.categories.length > 20) {
    window.emotionTrendState.categories.shift();
    window.emotionTrendState.valence.shift();
    window.emotionTrendState.arousal.shift();
  }

  refreshEmotionTrendChart();
}

function refreshEmotionTrendChart() {
  if (!window.emotionTrendChart || !window.emotionTrendState) {
    return;
  }

  window.emotionTrendChart.updateOptions({
    xaxis: {
      categories: window.emotionTrendState.categories,
      labels: {
        style: {
          colors: getLabelColor()
        }
      }
    }
  });

  window.emotionTrendChart.updateSeries([
    {
      name: 'Valence',
      data: window.emotionTrendState.valence
    },
    {
      name: 'Arousal',
      data: window.emotionTrendState.arousal
    }
  ]);
}

/* =========================================================
 * 人脸 / 视觉信息更新
 * ========================================================= */
function updateFaceInfo(face) {
  if (!face) {
    return;
  }

  setText('faceExpression', face.expression || '--');

  if (face.quality !== undefined) {
    setText('faceQuality', face.quality + '%');
    setText('faceQualityPercent', face.quality + '%');
  }

  if (face.detected !== undefined) {
    setText('faceDetected', face.detected ? 'Detected' : 'Not Detected');
  }
}

/* =========================================================
 * 外周指标更新
 * ========================================================= */
function updatePeripheralInfo(peripheral) {
  if (!peripheral) {
    return;
  }

  if (peripheral.hr !== undefined) {
    setText('hrValue', peripheral.hr === '--' ? '--' : peripheral.hr + ' bpm');
  }

  if (peripheral.hrv !== undefined) {
    setText('hrvValue', peripheral.hrv === '--' ? '--' : peripheral.hrv + ' ms');
  }

  if (peripheral.gsr !== undefined) {
    setText('gsrValue', peripheral.gsr === '--' ? '--' : peripheral.gsr + ' μS');
  }

  if (peripheral.resp !== undefined) {
    setText('respValue', peripheral.resp === '--' ? '--' : peripheral.resp + ' bpm');
  }
}

/* =========================================================
 * 视频本地预览
 * ========================================================= */
function bindTrialVideoPreview() {
  var input = document.getElementById('trialVideoInput');
  var video = document.getElementById('trialVideoPlayer');

  if (!input || !video) {
    console.warn('[Emotion Monitor] 视频预览控件未找到，请检查 HTML id', {
      input: input,
      video: video
    });
    return;
  }

  input.addEventListener('change', function () {
    var file = input.files[0];

    if (!file) {
      return;
    }

    console.log('[Emotion Monitor] 选择的视频文件：', file.name, file.type, file.size);

    var url = URL.createObjectURL(file);

    video.src = url;
    video.load();

    video.play().catch(function (err) {
      console.warn('[Emotion Monitor] 视频自动播放被浏览器阻止，需要手动点击播放：', err);
    });
  });
}

/* =========================================================
 * 工具函数
 * ========================================================= */
function setText(id, text) {
  var el = document.getElementById(id);

  if (el) {
    el.innerText = text;
  }
}

function formatNumber(x) {
  if (x === undefined || x === null || x === '--') {
    return '--';
  }

  var n = Number(x);

  if (isNaN(n)) {
    return '--';
  }

  return n.toFixed(2);
}

function formatPercent(x) {
  if (x === undefined || x === null || x === '--') {
    return '--';
  }

  var n = Number(x);

  if (isNaN(n)) {
    return '--';
  }

  return Math.round(n * 100) + '%';
}

function clampPercent(x) {
  x = Number(x || 0);

  if (x < 0) {
    return 0;
  }

  if (x > 100) {
    return 100;
  }

  return Math.round(x);
}

function createZeroData(length) {
  var data = [];

  for (var i = 0; i < length; i++) {
    data.push(0);
  }

  return data;
}

/* =========================================================
 * fallback 模拟信号
 * 仅当后端没有 raw_signals / signals 时使用
 * ========================================================= */
function updateSignalChartsByTrial(trialIndex) {
  trialIndex = Number(trialIndex || 1);

  updateEEGChartByTrial(trialIndex);
  updatePeripheralChartByTrial(trialIndex);
}

function updateEEGChartByTrial(trialIndex) {
  if (!window.eegCharts) {
    return;
  }

  var seed = trialIndex * 17;

  var fallbackSeries = [
    {
      name: 'Fp1',
      data: generateWaveDataWithSeed(300, 0.8 + trialIndex * 0.01, 0.25, seed + 1)
    },
    {
      name: 'F3',
      data: generateWaveDataWithSeed(300, 0.6 + trialIndex * 0.008, 0.35, seed + 2)
    },
    {
      name: 'F4',
      data: generateWaveDataWithSeed(300, 0.5 + trialIndex * 0.006, 0.4, seed + 3)
    },
    {
      name: 'Cz',
      data: generateWaveDataWithSeed(300, 0.7 + trialIndex * 0.005, 0.3, seed + 4)
    }
  ];

  fallbackSeries.forEach(function (item) {
    var chart = window.eegCharts[item.name];

    if (chart) {
      chart.updateSeries([
        {
          name: item.name,
          data: item.data
        }
      ]);
    }
  });
}

function updatePeripheralChartByTrial(trialIndex) {
  if (!window.peripheralChart) {
    return;
  }

  var seed = trialIndex * 31;

  window.peripheralChart.updateSeries([
    {
      name: 'GSR',
      data: generateSlowDataWithSeed(300, 0.4 + trialIndex * 0.005, seed + 1)
    },
    {
      name: 'Resp',
      data: generateWaveDataWithSeed(300, 0.8 + trialIndex * 0.007, 0.2, seed + 2)
    },
    {
      name: 'Pleth',
      data: generateWaveDataWithSeed(300, 0.9 + trialIndex * 0.007, 0.2, seed + 3)
    }
  ]);
}

function seededRandom(seed) {
  var x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function generateWaveDataWithSeed(length, amplitude, noise, seed) {
  var data = [];

  for (var i = 0; i < length; i++) {
    var r = seededRandom(seed + i * 13.37);

    var value =
      Math.sin((i + seed) / 5) * amplitude +
      Math.sin((i + seed) / 13) * amplitude * 0.5 +
      (r - 0.5) * noise;

    data.push(Number(value.toFixed(3)));
  }

  return data;
}

function generateSlowDataWithSeed(length, noise, seed) {
  var data = [];
  var value = 0.3 + seededRandom(seed) * 0.2;

  for (var i = 0; i < length; i++) {
    var r = seededRandom(seed + i * 7.91);
    value += (r - 0.5) * noise * 0.1;
    data.push(Number(value.toFixed(3)));
  }

  return data;
}

/* =========================================================
 * 旧模拟情绪函数
 * 保留但不主动调用
 * ========================================================= */
function updateEmotionMockData() {
  var valence = randomBetween(-0.2, 0.9);
  var arousal = randomBetween(-0.1, 0.95);
  var confidence = randomBetween(0.75, 0.98);

  updateEmotionCard(valence, arousal, confidence);

  if (window.vaChart) {
    window.vaChart.updateSeries([
      {
        name: 'Current Emotion',
        data: [[Number(valence.toFixed(2)), Number(arousal.toFixed(2))]]
      }
    ]);
  }
}

function updateEmotionCard(valence, arousal, confidence) {
  var emotionLabel = document.getElementById('emotionLabel');
  var emotionType = document.getElementById('emotionType');
  var valenceText = document.getElementById('valenceText');
  var arousalText = document.getElementById('arousalText');
  var confidenceText = document.getElementById('confidenceText');
  var valenceBar = document.getElementById('valenceBar');
  var arousalBar = document.getElementById('arousalBar');
  var confidenceBar = document.getElementById('confidenceBar');

  var emotion = classifyEmotion(valence, arousal);

  if (emotionLabel) {
    emotionLabel.innerText = emotion.label;
    emotionLabel.style.color = emotion.color;
  }

  if (emotionType) {
    emotionType.innerText = emotion.type;
  }

  if (valenceText) {
    valenceText.innerText = valence.toFixed(2);
  }

  if (arousalText) {
    arousalText.innerText = arousal.toFixed(2);
  }

  if (confidenceText) {
    confidenceText.innerText = Math.round(confidence * 100) + '%';
  }

  if (valenceBar) {
    valenceBar.style.width = clampPercent(((valence + 1) / 2) * 100) + '%';
  }

  if (arousalBar) {
    arousalBar.style.width = clampPercent(((arousal + 1) / 2) * 100) + '%';
  }

  if (confidenceBar) {
    confidenceBar.style.width = clampPercent(confidence * 100) + '%';
  }
}

function classifyEmotion(valence, arousal) {
  if (valence >= 0 && arousal >= 0) {
    return {
      label: 'Excited',
      type: '高效价 / 高唤醒',
      color: '#ffb020'
    };
  }

  if (valence >= 0 && arousal < 0) {
    return {
      label: 'Calm',
      type: '高效价 / 低唤醒',
      color: '#28c76f'
    };
  }

  if (valence < 0 && arousal >= 0) {
    return {
      label: 'Tense',
      type: '低效价 / 高唤醒',
      color: '#ea5455'
    };
  }

  return {
    label: 'Sad',
    type: '低效价 / 低唤醒',
    color: '#7367f0'
  };
}

function randomBetween(min, max) {
  return Math.random() * (max - min) + min;
}
