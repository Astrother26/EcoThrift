document.addEventListener("DOMContentLoaded", function () {
    const uploadForm  = document.getElementById("uploadForm");
    const imageInput  = document.getElementById("imageInput");
    const resultsDiv  = document.getElementById("results");

    // Inject gender selector before the submit button
    const submitBtn = uploadForm.querySelector('button[type="submit"]') 
                   || uploadForm.querySelector('input[type="submit"]');

    if (submitBtn && !document.getElementById('genderSelect')) {
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;';

        const label = document.createElement('label');
        label.textContent = 'Section: ';
        label.style.fontWeight = '600';

        const select = document.createElement('select');
        select.id = 'genderSelect';
        select.style.cssText = 'padding:0.5rem 1rem;border-radius:8px;border:2px solid #56C596;font-size:0.9rem;';

        [
            { value: '',      label: 'All' },
            { value: 'woman', label: 'Women' },
            { value: 'man',   label: 'Men'   }
        ].forEach(function(opt) {
            const o     = document.createElement('option');
            o.value     = opt.value;
            o.textContent = opt.label;
            select.appendChild(o);
        });

        wrapper.appendChild(label);
        wrapper.appendChild(select);
        submitBtn.parentNode.insertBefore(wrapper, submitBtn);
    }

    uploadForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        const file = imageInput.files[0];

        if (!file) {
            alert("Please upload an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        // Send gender filter
        const gender = document.getElementById('genderSelect')?.value || '';
        if (gender) formData.append('gender', gender);

        resultsDiv.innerHTML = "<p>🔍 Analyzing image... please wait.</p>";

        try {
            const response = await fetch("http://127.0.0.1:5000/api/recommend", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) throw new Error(`Server returned ${response.status}`);

            const data = await response.json();

            if (data.error) {
                resultsDiv.innerHTML = `<p style="color:red;">⚠️ ${data.error}</p>`;
            } else if (data.recommendations) {
                resultsDiv.innerHTML = `
                    <h3>✅ Recommendations (${data.count}):</h3>
                    <ul>
                        ${data.recommendations.map(p =>
                            `<li><strong>${p.name || 'Unnamed'}</strong> 
                             — ${p.gender || ''} 
                             — Carbon: ${parseFloat(p.carbon_kg||0).toFixed(1)} kg
                             — Score: ${Math.round((p.hybrid_score||0)*100)}%</li>`
                        ).join("")}
                    </ul>`;
            }
        } catch (error) {
            console.error("Fetch error:", error);
            resultsDiv.innerHTML = `<p style="color:red;">❌ Failed to fetch recommendations.</p>`;
        }
    });
});