// 等待DOM内容加载完成
document.addEventListener('DOMContentLoaded', function() {
    // 获取搜索输入框和开始按钮元素
    const searchInput = document.getElementById('search-input');
    const startButton = document.getElementById('start-button');
    const resultContainer = document.getElementById('result-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const chartContainer = document.getElementById('chart-container');
    const analysisContainer = document.getElementById('analysis-container');
    const errorContainer = document.getElementById('error-container');
    
    // 处理搜索按钮点击事件
    startButton.addEventListener('click', function() {
        processQuery();
    });
    
    // 处理回车键按下事件
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            processQuery();
        }
    });
    
    // 处理用户查询
    function processQuery() {
        const query = searchInput.value.trim();
        
        if (!query) {
            showError('请输入搜索内容');
            return;
        }
        
        // 显示加载指示器
        showLoading();
        
        // 清空之前的结果
        clearResults();
        
        // 发送API请求
        fetch('/api/process_query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        })
        .then(response => response.json())
        .then(data => {
            // 隐藏加载指示器
            hideLoading();
            
            if (data.success) {
                // 显示图表
                displayChart(data.chart_image);
                
                // 显示分析结果
                displayAnalysis(data.analysis, data.chart_title, data.chart_type);
                
                // 显示结果容器
                resultContainer.style.display = 'block';
                
                // 确保搜索框仍然可见
                document.querySelector('.search-container').style.display = 'flex';
                
                // 滚动到结果区域
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                // 显示错误信息
                showError(data.error || '处理查询时出错');
            }
        })
        .catch(error => {
            // 隐藏加载指示器
            hideLoading();
            
            // 显示错误信息
            showError('网络请求失败: ' + error.message);
            console.error('Error:', error);
        });
    }
    
    // 显示图表
    function displayChart(chartImageBase64) {
        // 创建图片元素
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + chartImageBase64;
        img.alt = '数据可视化图表';
        img.className = 'chart-image';
        
        // 清空图表容器并添加图片
        chartContainer.innerHTML = '';
        chartContainer.appendChild(img);
    }
    
    // 显示分析结果
    function displayAnalysis(analysis, title, chartType) {
        // 创建标题元素
        const titleElement = document.createElement('h2');
        titleElement.textContent = title;
        titleElement.className = 'analysis-title';
        
        // 创建图表类型元素
        const chartTypeElement = document.createElement('p');
        chartTypeElement.textContent = '图表类型: ' + getChartTypeName(chartType);
        chartTypeElement.className = 'chart-type';
        
        // 创建分析文本元素
        const analysisElement = document.createElement('p');
        analysisElement.textContent = analysis;
        analysisElement.className = 'analysis-text';
        
        // 清空分析容器并添加元素
        analysisContainer.innerHTML = '';
        analysisContainer.appendChild(titleElement);
        analysisContainer.appendChild(chartTypeElement);
        analysisContainer.appendChild(analysisElement);
    }
    
    // 获取图表类型的中文名称
    function getChartTypeName(chartType) {
        const chartTypeMap = {
            'bar': '柱状图',
            'line': '折线图',
            'pie': '饼图',
            'scatter': '散点图'
        };
        
        return chartTypeMap[chartType] || chartType;
    }
    
    // 显示错误信息
    function showError(message) {
        errorContainer.textContent = message;
        errorContainer.style.display = 'block';
        resultContainer.style.display = 'none';
    }
    
    // 清空结果
    function clearResults() {
        chartContainer.innerHTML = '';
        analysisContainer.innerHTML = '';
        errorContainer.style.display = 'none';
    }
    
    // 显示加载指示器
    function showLoading() {
        loadingIndicator.style.display = 'flex';
    }
    
    // 隐藏加载指示器
    function hideLoading() {
        loadingIndicator.style.display = 'none';
    }
});
