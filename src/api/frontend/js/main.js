/**
 * EcoThrift Frontend — main.js
 * localStorage-first cart. Flask API used as bonus sync only.
 *
 * Changes vs original:
 * - handleUpload: reads gender selector and appends to FormData
 * - setupEventListeners: auto-detect gender from filename as fallback
 * - createProductCard: shows gender badge on each card
 */

const API_URL  = 'http://127.0.0.1:5000';
const CART_KEY  = 'et_cart';
const STATS_KEY = 'et_stats';

let uploadInput, uploadBtn, resultsContainer, loadingIndicator, errorMessage;

/* ================================================================
   LOCALSTORAGE CART HELPERS
================================================================ */
function getCart() {
    try { return JSON.parse(localStorage.getItem(CART_KEY) || '[]'); }
    catch(e) { return []; }
}

function saveCart(cart) {
    localStorage.setItem(CART_KEY, JSON.stringify(cart));
    refreshCartBadge(cart);
}

function refreshCartBadge(cart) {
    cart = cart || getCart();
    const total = cart.reduce((s, i) => s + (i.quantity || 1), 0);
    const badge = document.getElementById('cart-count');
    if (badge) {
        badge.textContent = total;
        badge.style.display = total > 0 ? 'block' : 'none';
    }
}

/* ================================================================
   INIT
================================================================ */
document.addEventListener('DOMContentLoaded', function () {
    console.log('🌿 EcoThrift initialized');

    uploadInput      = document.getElementById('imageUpload');
    uploadBtn        = document.getElementById('uploadBtn');
    resultsContainer = document.getElementById('results');
    loadingIndicator = document.getElementById('loading');
    errorMessage     = document.getElementById('errorMessage');

    injectGenderSelector();   // Add gender dropdown near upload button
    setupEventListeners();
    refreshCartBadge();
    tryAPIHealth();
});

/* ================================================================
   GENDER SELECTOR — injected next to the upload button
================================================================ */
function injectGenderSelector() {
    // Don't inject twice
    if (document.getElementById('genderSelect')) return;

    const btn = document.getElementById('uploadBtn');
    if (!btn) return;

    // Create wrapper so selector sits inline with the button
    const wrapper = document.createElement('div');
    wrapper.id    = 'gender-wrapper';
    wrapper.style.cssText =
        'display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;justify-content:center;margin-top:0.5rem;';

    // Gender dropdown
    const select = document.createElement('select');
    select.id    = 'genderSelect';
    select.style.cssText =
        'padding:0.6rem 1rem;border-radius:8px;border:2px solid #56C596;' +
        'background:#fff;color:#2D5016;font-size:0.9rem;font-weight:600;cursor:pointer;' +
        'outline:none;appearance:none;-webkit-appearance:none;' +
        'background-image:url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'12\' height=\'8\'%3E%3Cpath d=\'M1 1l5 5 5-5\' stroke=\'%2356C596\' stroke-width=\'2\' fill=\'none\'/%3E%3C/svg%3E");' +
        'background-repeat:no-repeat;background-position:right 0.75rem center;padding-right:2rem;';

    [
        { value: '',       label: '👕  All' },
        { value: 'woman',  label: '👗  Women' },
        { value: 'man',    label: '👔  Men' }
    ].forEach(function(opt) {
        const o   = document.createElement('option');
        o.value   = opt.value;
        o.textContent = opt.label;
        select.appendChild(o);
    });

    // Move button into wrapper, add select before it
    btn.parentNode.insertBefore(wrapper, btn);
    wrapper.appendChild(select);
    wrapper.appendChild(btn);

    console.log('✅ Gender selector injected');
}

function getSelectedGender() {
    const sel = document.getElementById('genderSelect');
    return sel ? sel.value : '';
}

/* ================================================================
   EVENT LISTENERS
================================================================ */
function setupEventListeners() {
    if (uploadBtn) uploadBtn.addEventListener('click', handleUpload);

    if (uploadInput) {
        uploadInput.addEventListener('change', function () {
            if (this.files && this.files[0]) {
                const h3 = document.querySelector('.upload-area h3');
                if (h3) h3.textContent = 'Selected: ' + this.files[0].name;

                // Auto-detect gender from filename as a convenience hint
                // e.g. "dress_photo.jpg" → set selector to 'woman'
                autoDetectGenderFromFilename(this.files[0].name);
            }
        });
    }

    const uploadArea = document.querySelector('.upload-area');
    if (uploadArea) {
        uploadArea.addEventListener('click', function (e) {
            if (e.target !== uploadInput) uploadInput.click();
        });
    }
}

function autoDetectGenderFromFilename(filename) {
    const lower  = filename.toLowerCase();
    const select = document.getElementById('genderSelect');
    if (!select) return;

    const womanHints = ['dress', 'skirt', 'blouse', 'womens', 'woman', 'girl', 'ladies', 'bodysuit'];
    const manHints   = ['mens', 'man', 'trunks', 'boxer', 'suit'];

    if (womanHints.some(w => lower.includes(w))) {
        select.value = 'woman';
        showNotification('Auto-detected: Women\'s item', 'info');
    } else if (manHints.some(w => lower.includes(w))) {
        select.value = 'man';
        showNotification('Auto-detected: Men\'s item', 'info');
    }
    // If no hint found, leave selector as-is
}

async function tryAPIHealth() {
    try {
        await fetch(API_URL + '/api/health');
        console.log('✅ API reachable');
    } catch (e) {
        console.warn('⚠️ API not reachable — using localStorage only');
    }
}

/* ================================================================
   HANDLE IMAGE UPLOAD & RECOMMENDATIONS
================================================================ */
async function handleUpload() {
    if (!uploadInput || !uploadInput.files || !uploadInput.files.length) {
        showError('Please select an image file first'); return;
    }

    const file = uploadInput.files[0];
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Please select a valid image file (PNG, JPG, GIF, WEBP)'); return;
    }
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB'); return;
    }

    showLoading(); hideError(); clearResults();

    const formData = new FormData();
    formData.append('image', file);

    // ── Gender filter ────────────────────────────────────────────
    const gender = getSelectedGender();
    if (gender) {
        formData.append('gender', gender);
        console.log('🔍 Gender filter applied:', gender);
    } else {
        console.log('🔍 No gender filter — showing all');
    }
    // ────────────────────────────────────────────────────────────

    try {
        const response = await fetch(API_URL + '/api/recommend', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || ('Server error: ' + response.status));
        }

        const data = await response.json();

        if (!data.success || !data.recommendations || !data.recommendations.length) {
            showError('No recommendations found. Try uploading a different clothing image.');
            return;
        }

        displayRecommendations(data.recommendations);
        setTimeout(function() {
            if (resultsContainer) resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);

    } catch (error) {
        console.error('❌ Recommend error:', error);
        showError('Failed to get recommendations: ' + error.message);
    } finally {
        hideLoading();
    }
}

/* ================================================================
   DISPLAY RECOMMENDATIONS
================================================================ */
function displayRecommendations(recommendations) {
    if (!resultsContainer) return;
    resultsContainer.innerHTML = '';
    recommendations.forEach(function(product, index) {
        resultsContainer.appendChild(createProductCard(product, index));
    });

    if (typeof injectQuickViewButtons === 'function') {
        injectQuickViewButtons(recommendations);
    }
}

function createProductCard(product, index) {
    const card = document.createElement('div');
    card.className = 'product-card';
    card.style.animation      = 'fadeInUp 0.5s ease-out forwards';
    card.style.animationDelay = (index * 0.1) + 's';
    card.style.opacity        = '0';

    const placeholder = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='500'%3E%3Crect width='400' height='500' fill='%23f3f3f3'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' font-size='20' fill='%239ca3af'%3ENo Image%3C/text%3E%3C/svg%3E";
    const img       = product.image_url || product.image || '';
    const grade     = product.sustainability_grade || 'F';
    const matchPct  = Math.round((product.similarity_score || product.hybrid_score || 0) * 100);
    const ecoScore  = Math.round(product.sustainability_score || 0);
    const carbonKg  = parseFloat(product.carbon_kg  || 0).toFixed(1);
    const savingsKg = parseFloat(product.savings_kg || 4.8).toFixed(1);
    const priceINR  = ((product.price || 0) * 83).toFixed(2);

    // Gender badge — only show if present and not unisex
    const genderVal   = (product.gender || '').toLowerCase();
    const genderBadge = (genderVal === 'woman')
        ? '<span style="font-size:0.75rem;background:#fce4ec;color:#c2185b;padding:2px 8px;border-radius:12px;font-weight:600;">Women</span>'
        : (genderVal === 'man')
        ? '<span style="font-size:0.75rem;background:#e3f2fd;color:#1565c0;padding:2px 8px;border-radius:12px;font-weight:600;">Men</span>'
        : '';

    const productJson = JSON.stringify(product).replace(/"/g, '&quot;');

    card.innerHTML =
        '<div class="product-image">' +
            '<img src="' + (img || placeholder) + '" alt="' + (product.name || 'Product') + '"' +
                ' onerror="this.src=\'' + placeholder + '\';" loading="lazy">' +
            '<div class="eco-badge grade-' + grade + '">Grade ' + grade + '</div>' +
        '</div>' +
        '<div class="product-info">' +
            '<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;">' +
                '<h3 class="product-name" style="margin:0;flex:1;">' + (product.name || 'Unknown Product') + '</h3>' +
                genderBadge +
            '</div>' +
            '<p style="color:#666;margin-bottom:0.5rem;font-size:0.9rem;text-transform:capitalize;">' +
                (product.brand || 'Unknown Brand') + '</p>' +
            '<div class="product-price">&#8377;' + priceINR + '</div>' +
            '<div class="eco-metrics">' +
                '<div class="metric"><span class="metric-label">Match</span><div class="metric-value">' + matchPct + '%</div></div>' +
                '<div class="metric"><span class="metric-label">Eco Score</span><div class="metric-value">' + ecoScore + '%</div></div>' +
                '<div class="metric"><span class="metric-label">Carbon</span><div class="metric-value">' + carbonKg + ' kg</div></div>' +
            '</div>' +
            '<div class="environmental-savings">' +
                '<div class="savings-item"><i class="fas fa-tint"></i><span>' + parseFloat(product.water_liters || 0).toFixed(0) + 'L water</span></div>' +
                '<div class="savings-item"><i class="fas fa-bolt"></i><span>' + parseFloat(product.energy_mj   || 0).toFixed(0) + 'MJ energy</span></div>' +
                '<div class="savings-item"><i class="fas fa-leaf"></i><span>Saves ' + savingsKg + ' kg CO&#x2082;</span></div>' +
            '</div>' +
            '<button class="add-to-cart-btn" onclick="addToCart(' + productJson + ')">' +
                '<i class="fas fa-shopping-bag"></i> Add to Cart' +
            '</button>' +
        '</div>';

    return card;
}

/* ================================================================
   ADD TO CART — localStorage primary, API secondary
================================================================ */
function addToCart(productOrSku, name, price, carbonKg, savingsKg) {
    var item;

    if (typeof productOrSku === 'object' && productOrSku !== null) {
        var p = productOrSku;
        item = {
            sku:                  p.sku || p.product_id || ('item_' + Date.now()),
            name:                 p.name || 'Unknown Product',
            price:                parseFloat(p.price || 0),
            carbon_kg:            parseFloat(p.carbon_kg || 0),
            water_liters:         parseFloat(p.water_liters || 0),
            energy_mj:            parseFloat(p.energy_mj || 0),
            savings_kg:           parseFloat(p.savings_kg || 4.8),
            image_url:            p.image_url || p.image || '',
            sustainability_grade: p.sustainability_grade || 'C',
            sustainability_score: parseFloat(p.sustainability_score || 0),
            brand:                p.brand || '',
            description:          p.description || '',
            gender:               p.gender || '',
            quantity:             1
        };
    } else {
        item = {
            sku:        productOrSku || ('item_' + Date.now()),
            name:       name || 'Product',
            price:      parseFloat(price || 0),
            carbon_kg:  parseFloat(carbonKg || 0),
            savings_kg: parseFloat(savingsKg || 4.8),
            quantity:   1
        };
    }

    // Save to localStorage
    var cart = getCart();
    var existing = null;
    for (var i = 0; i < cart.length; i++) {
        if (cart[i].sku === item.sku) { existing = cart[i]; break; }
    }
    if (existing) {
        existing.quantity = (existing.quantity || 1) + 1;
    } else {
        cart.push(item);
    }
    saveCart(cart);

    // Button feedback
    var btn = null;
    if (event && event.target) {
        btn = event.target.closest('.add-to-cart-btn') || event.target.closest('.modal-cart-btn');
    }
    if (btn) {
        var orig   = btn.innerHTML;
        var origBg = btn.style.background;
        btn.innerHTML        = '<i class="fas fa-check"></i> Added!';
        btn.style.background = 'linear-gradient(135deg, #4CAF50, #8BC34A)';
        btn.disabled         = true;
        setTimeout(function() {
            btn.innerHTML        = orig;
            btn.style.background = origBg;
            btn.disabled         = false;
        }, 2000);
    }

    showNotification('Added "' + item.name + '" to cart!', 'success');
    updateProfileStats(item);
    tryAPICart(item);
}

function updateProfileStats(item) {
    try {
        var stats = JSON.parse(localStorage.getItem(STATS_KEY) || '{}');
        stats.totalItems  = (stats.totalItems  || 0) + 1;
        stats.totalCarbon = parseFloat(((stats.totalCarbon || 0) + (item.savings_kg   || 4.8)).toFixed(2));
        stats.totalWater  = parseFloat(((stats.totalWater  || 0) + (item.water_liters || 2700)).toFixed(0));
        stats.totalEnergy = parseFloat(((stats.totalEnergy || 0) + (item.energy_mj    || 0)).toFixed(1));
        stats.level       = Math.floor(stats.totalItems / 5) + 1;

        stats.achievements = stats.achievements || [];
        if (stats.totalItems >= 1  && stats.achievements.indexOf('first_purchase') === -1) stats.achievements.push('first_purchase');
        if (stats.totalItems >= 5  && stats.achievements.indexOf('eco_warrior')    === -1) stats.achievements.push('eco_warrior');
        if (stats.totalItems >= 10 && stats.achievements.indexOf('planet_saver')   === -1) stats.achievements.push('planet_saver');
        if (stats.totalItems >= 20 && stats.achievements.indexOf('eco_master')     === -1) stats.achievements.push('eco_master');
        if (stats.totalCarbon >= 10 && stats.achievements.indexOf('carbon_hero')   === -1) stats.achievements.push('carbon_hero');

        localStorage.setItem(STATS_KEY, JSON.stringify(stats));
        console.log('📊 Stats updated:', stats);
    } catch(e) {
        console.error('Stats update error:', e);
    }
}

async function tryAPICart(item) {
    try {
        var userEmail = 'user@ecothrift.com';
        try { userEmail = JSON.parse(localStorage.getItem('user') || '{}').email || userEmail; } catch(e) {}
        await fetch(API_URL + '/api/cart/' + userEmail, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(item)
        });
    } catch (e) { /* silent */ }
}

/* ================================================================
   UI HELPERS
================================================================ */
function showLoading() {
    if (loadingIndicator) loadingIndicator.style.display = 'block';
    if (uploadBtn) {
        uploadBtn.disabled   = true;
        uploadBtn.innerHTML  = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    }
}

function hideLoading() {
    if (loadingIndicator) loadingIndicator.style.display = 'none';
    if (uploadBtn) {
        uploadBtn.disabled  = false;
        uploadBtn.innerHTML = '<i class="fas fa-magic"></i> Get Recommendations';
    }
}

function showError(message) {
    if (errorMessage) {
        errorMessage.textContent   = '❌ ' + message;
        errorMessage.style.display = 'block';
    }
    console.error('❌', message);
}

function hideError()   { if (errorMessage)     errorMessage.style.display   = 'none'; }
function clearResults() { if (resultsContainer) resultsContainer.innerHTML = '';      }

function showNotification(message, type) {
    type = type || 'info';
    var n = document.createElement('div');
    n.style.cssText =
        'position:fixed;top:80px;right:20px;' +
        'padding:1rem 1.5rem;' +
        'background:' + (type === 'success' ? '#2D5016' : type === 'info' ? '#1565c0' : '#f44336') + ';' +
        'color:white;border-radius:12px;' +
        'box-shadow:0 4px 20px rgba(0,0,0,0.2);' +
        'z-index:10000;font-weight:600;font-size:0.92rem;' +
        'animation:slideInRight 0.3s ease-out;' +
        'max-width:320px;';
    n.textContent = message;
    document.body.appendChild(n);
    setTimeout(function() {
        n.style.opacity    = '0';
        n.style.transition = 'opacity 0.3s';
        setTimeout(function() { n.remove(); }, 300);
    }, 3000);
}

function filterCategory(category) {
    var aiSection = document.querySelector('.ai-section');
    if (aiSection) aiSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    showNotification('Upload an image to find ' + category, 'info');
}

/* ================================================================
   STYLES
================================================================ */
var style = document.createElement('style');
style.textContent =
    '.environmental-savings{display:flex;flex-direction:column;gap:0.4rem;margin:0.8rem 0;padding:0.9rem;background:linear-gradient(135deg,#E8F5E9,#C8E6C9);border-radius:12px;border-left:4px solid #56C596;}' +
    '.savings-item{display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:#2D5016;font-weight:500;}' +
    '.savings-item i{color:#56C596;width:18px;text-align:center;}' +
    '#genderSelect:focus{border-color:#2D5016;box-shadow:0 0 0 3px rgba(86,197,150,0.2);}' +
    '#gender-wrapper{margin-top:1rem;}' +
    '@keyframes fadeInUp{from{opacity:0;transform:translateY(30px);}to{opacity:1;transform:translateY(0);}}' +
    '@keyframes slideInRight{from{transform:translateX(350px);opacity:0;}to{transform:translateX(0);opacity:1;}}';
document.head.appendChild(style);

/* ================================================================
   GLOBAL EXPORTS
================================================================ */
window.addToCart        = addToCart;
window.filterCategory   = filterCategory;
window.getCart          = getCart;
window.saveCart         = saveCart;
window.refreshCartBadge = refreshCartBadge;

console.log('✅ EcoThrift main.js loaded — gender filter active');