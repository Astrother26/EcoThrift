import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const EMISSION_FACTORS: Record<string, { co2: number; water: number; energy: number; grade: string }> = {
  cotton: { co2: 7.5, water: 2967, energy: 158, grade: "B" },
  organic_cotton: { co2: 4.1, water: 254, energy: 60, grade: "A" },
  recycled_cotton: { co2: 1.5, water: 148, energy: 47, grade: "A+" },
  linen: { co2: 3.4, water: 738, energy: 184, grade: "A" },
  hemp: { co2: 1.78, water: 1.46, energy: 2.1, grade: "A+" },
  wool: { co2: 13.0, water: 272, energy: 103, grade: "D" },
  silk: { co2: 11.5, water: 1000, energy: 140, grade: "D" },
  leather: { co2: 110, water: 17000, energy: 200, grade: "F" },
  polyester: { co2: 4.4, water: 26, energy: 97, grade: "C" },
  recycled_polyester: { co2: 1.2, water: 5, energy: 13, grade: "A+" },
  nylon: { co2: 10.2, water: 40, energy: 150, grade: "D" },
  acrylic: { co2: 9.5, water: 120, energy: 150, grade: "D" },
  viscose: { co2: 5.0, water: 400, energy: 120, grade: "C" },
  lyocell: { co2: 3.0, water: 200, energy: 80, grade: "A" },
};

const GARMENT_WEIGHTS: Record<string, number> = {
  tshirt: 0.17, "t-shirt": 0.17, tee: 0.17,
  shirt: 0.23, blouse: 0.23,
  jeans: 0.79, denim: 0.79, trousers: 0.56, pants: 0.56,
  dress: 0.475, skirt: 0.34,
  jacket: 0.68, coat: 1.68, blazer: 0.68,
  sweater: 0.45, hoodie: 0.90, sweatshirt: 0.90,
  shorts: 0.20,
  default: 0.40,
};

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const { material, category, weight_kg } = await req.json();
    if (!material || !category) {
      return new Response(JSON.stringify({ error: "material and category are required" }), {
        status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const mat = material.toLowerCase().replace(/\s+/g, "_");
    const factors = EMISSION_FACTORS[mat] || EMISSION_FACTORS["cotton"];
    const weight = weight_kg || GARMENT_WEIGHTS[category.toLowerCase()] || GARMENT_WEIGHTS["default"];

    const carbon_kg = parseFloat((factors.co2 * weight).toFixed(2));
    const water_liters = parseFloat((factors.water * weight).toFixed(0));
    const energy_mj = parseFloat((factors.energy * weight).toFixed(0));
    const score = Math.max(0, Math.min(100, 100 - carbon_kg * 5));

    // Savings vs buying new (average new = 4x thrift impact)
    const savings_carbon = parseFloat((carbon_kg * 3).toFixed(2));
    const savings_water = parseFloat((water_liters * 3).toFixed(0));

    return new Response(JSON.stringify({
      success: true,
      carbon_kg,
      water_liters,
      energy_mj,
      sustainability_grade: factors.grade,
      sustainability_score: Math.round(score),
      savings_vs_new: {
        carbon_saved_kg: savings_carbon,
        water_saved_l: savings_water,
        percentage: 75,
      },
      breakdown: {
        material_used: mat,
        garment_weight_kg: weight,
        co2_per_kg: factors.co2,
        water_per_kg: factors.water,
      },
    }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("carbon-calculate error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
