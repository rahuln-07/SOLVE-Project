// ============================================================
// Project WISE - Model 2 Prototype (GEE Pipeline)
// File: code/gee_pipeline.js
//
// Purpose:
// 1) Build a multi-layer geospatial composite (NDVI, slope, TWI, drainage density,
//    rainfall, LULC, etc.) for a chosen administrative region.
// 2) Create proxy labels using NDVI percentile thresholds (cropland-first).
// 3) Sample positive/negative points, extract PATCH neighborhood arrays,
//    export as batched TFRecords to Google Drive.
//
// Notes:
// - Labels in this prototype are proxy NDVI-based labels; replace with real
//   well/check-dam ground truth in the next phase.
// ============================================================

// -------------------- REGION (placeholder example) --------------------
var district = ee.FeatureCollection("FAO/GAUL/2015/level2")
  .filter(ee.Filter.eq('ADM2_NAME', 'Anantapur'));

Map.centerObject(district, 9);

// -------------------- CONFIG --------------------
var START = '2023-04-01';
var END = '2023-06-15';
var CLOUDY_PIXEL_PERCENTAGE = 60;

var NUM_POS = 2000;
var NUM_NEG = 2000;
var CANDIDATE_POOL = 50000;

var PATCH = 65;                 // odd => neighborhood kernel
var TARGET_SCALE = 10;          // meters
var BATCH_SIZE = 500;           // TFRecord batch size per export task
var TILE_SCALE = 4;             // increase if sampleRegions OOM (8 or 16)

var EXPORT_FOLDER = 'ProjectWise_Data';

// -------------------- helper --------------------
function p(label, v){ print(label, v); }

// -------------------- S2 masking + NDVI --------------------
function maskS2SR(img){
  var sclExists = img.bandNames().contains('SCL');

  // If SCL exists, mask cloud/shadow classes.
  var sclMask = ee.Image(ee.Algorithms.If(
    sclExists,
    img.select('SCL')
      .neq(8)  // cloud medium probability
      .and(img.select('SCL').neq(9))   // cloud high probability
      .and(img.select('SCL').neq(10)), // thin cirrus
    ee.Image(1)
  ));

  // QA60 bitmask fallback
  var qa = img.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var qaMask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  var combined = ee.Image(ee.Algorithms.If(sclExists, sclMask, qaMask));
  return img.updateMask(combined);
}

var s2raw = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(district)
  .filterDate(START, END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUDY_PIXEL_PERCENTAGE));

var s2masked = s2raw.map(maskS2SR);

p('s2raw.size()', s2raw.size());
p('s2masked.size()', s2masked.size());

function perImageNdvi(img){
  return img.normalizedDifference(['B8','B4'])
    .rename('ndvi')
    .copyProperties(img, ['system:time_start']);
}

var ndvi_raw_col = s2raw.map(perImageNdvi);
var ndvi_masked_col = s2masked.map(perImageNdvi);

var ndvi_raw_med = ndvi_raw_col.median().rename('ndvi');
var ndvi_masked_med = ndvi_masked_col.median().rename('ndvi');

// -------------------- other inputs --------------------
var srtm = ee.Image("USGS/SRTMGL1_003").clip(district);
var slope = ee.Terrain.slope(srtm).rename('slope');

var hydroSheds = ee.Image("WWF/HydroSHEDS/15ACC").clip(district);
var flowAcc = hydroSheds.select([0]).rename('flow_acc');

var worldCover = ee.Image("ESA/WorldCover/v200/2021").clip(district);
var lulc = worldCover.select('Map').rename('lulc');

var rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
  .filterDate('2022-06-01','2023-05-31')
  .sum()
  .rename('rainfall')
  .clip(district);

// -------------------- build composite --------------------
var targetProj = ndvi_masked_med.projection();
var scale = TARGET_SCALE;

var ndvi = ndvi_masked_med.resample('bilinear').reproject({crs: targetProj, scale: scale});
slope = slope.resample('bilinear').reproject({crs: targetProj, scale: scale});
rainfall = rainfall.resample('bilinear').reproject({crs: targetProj, scale: scale});
flowAcc = flowAcc.resample('bilinear').reproject({crs: targetProj, scale: scale});
lulc = lulc.reproject({crs: targetProj, scale: scale});

// Streams mask + drainage density
var streams = flowAcc.gt(1000).selfMask().reproject({crs: targetProj, scale: scale});
var ddKernel = ee.Kernel.circle({radius:1000, units:'meters', normalize:false});
var drainageDensity = streams.unmask(0)
  .reduceNeighborhood({reducer: ee.Reducer.sum(), kernel: ddKernel})
  .rename('drainage_density');

// Topographic Wetness Index (TWI) (basic approximation)
var twi = flowAcc.add(1e-6).expression(
  'log(acc / tan(slope))',
  {
    'acc': flowAcc.add(1e-6),
    'slope': slope.multiply(Math.PI/180).max(0.001)
  }
).rename('twi');

var composite = ee.Image.cat([
    slope.rename('slope'),
    twi.rename('twi'),
    drainageDensity.rename('drainage_density'),
    lulc.rename('lulc'),
    rainfall.rename('rainfall'),
    ndvi.rename('ndvi')
  ])
  .toFloat()
  .reproject({crs: targetProj, scale: scale});

p('Composite band names:', composite.bandNames());

// -------------------- proxy label thresholds (p25/p75, cropland-first) --------------------
var croplandMask = lulc.eq(40);

// helper: percentiles over a mask (client-side via evaluate)
function getPercentilesOverMask(maskImg, cb){
  var masked = ndvi.updateMask(maskImg);

  var perc = masked.reduceRegion({
    reducer: ee.Reducer.percentile([25,75]),
    geometry: district.geometry(),
    scale: scale,
    maxPixels: 1e13
  });

  perc.evaluate(function(d){
    if (!d) { cb(null); return; }

    // collect numeric values from the returned dict
    var vals = [];
    for (var k in d) {
      if (d.hasOwnProperty(k) && typeof d[k] === 'number') vals.push(d[k]);
    }

    if (vals.length >= 2){
      vals.sort(function(a,b){return a-b;});
      cb({p25: vals[0], p75: vals[1]});
    } else {
      cb(null);
    }
  });
}

getPercentilesOverMask(croplandMask, function(croplPerc){
  if (croplPerc !== null) {
    p('Using cropland percentiles (p25,p75):', croplPerc.p25, croplPerc.p75);
    proceedWithThresholds(croplPerc.p25, croplPerc.p75);
  } else {
    // fallback to entire district
    getPercentilesOverMask(ee.Image(1), function(allPerc){
      if (allPerc !== null){
        p('Using district percentiles (p25,p75):', allPerc.p25, allPerc.p75);
        proceedWithThresholds(allPerc.p25, allPerc.p75);
      } else {
        p('Percentiles unavailable, fallback thresholds used');
        proceedWithThresholds(0.12, 0.22);
      }
    });
  }
});

// -------------------- sampling & batching --------------------
function proceedWithThresholds(p25, p75){
  p25 = Number(p25);
  p75 = Number(p75);

  if (isNaN(p25) || isNaN(p75)){
    p25 = 0.12;
    p75 = 0.22;
  }

  // guard against degenerate percentiles
  if (p25 >= p75){
    var mid = p25;
    p25 = Math.max(mid - 0.03, -1);
    p75 = Math.min(mid + 0.03, 1);
  }

  var POS_NDVI = p75;
  var NEG_NDVI = p25;

  p('Chosen thresholds -> NEG_NDVI (p25):', NEG_NDVI, 'POS_NDVI (p75):', POS_NDVI);

  var class1_mask = croplandMask.and(ndvi.gt(POS_NDVI));
  var class0_mask = croplandMask.and(ndvi.lt(NEG_NDVI));
  var class1_ndvi_only = ndvi.gt(POS_NDVI);
  var class0_ndvi_only = ndvi.lt(NEG_NDVI);

  function samplePool(maskImg, requested, label){
    return maskImg.selfMask().sample({
        region: district,
        scale: scale,
        numPixels: Math.min(CANDIDATE_POOL, Math.max(requested,10000)),
        geometries: true,
        seed: Math.floor(Math.random() * 1e6)
      })
      .randomColumn('rand')
      .sort('rand')
      .limit(requested)
      .map(function(f){ return f.set('label', label); });
  }

  var posFromCropl = samplePool(class1_mask, NUM_POS, 1);
  var negFromCropl = samplePool(class0_mask, NUM_NEG, 0);

  // evaluate sizes then supplement if needed
  posFromCropl.size().evaluate(function(npos){
    negFromCropl.size().evaluate(function(nneg){
      p('Initial cropland-sampled (pos,neg):', npos, nneg);

      var needPos = Math.max(NUM_POS - npos, 0);
      var needNeg = Math.max(NUM_NEG - nneg, 0);

      var posFC = posFromCropl;
      var negFC = negFromCropl;

      if (needPos > 0) {
        var supPos = samplePool(class1_ndvi_only, needPos, 1);
        posFC = posFC.merge(supPos);
      }
      if (needNeg > 0) {
        var supNeg = samplePool(class0_ndvi_only, needNeg, 0);
        negFC = negFC.merge(supNeg);
      }

      posFC.size().evaluate(function(finalPos){
        negFC.size().evaluate(function(finalNeg){
          p('Final sample sizes (pos,neg):', finalPos, finalNeg);

          if (finalPos === 0 || finalNeg === 0){
            p('ERROR: insufficient samples after fallback. Consider wider window / relax cloud filter / lower thresholds.');
            return;
          }

          // Merge and add a random column for partitioning into batches
          var allPoints = posFC.merge(negFC).randomColumn('rand_final');
          allPoints = allPoints.select(['.*']); // keep geometry + props

          // Precompute patch array image once
          var radiusPixels = Math.floor((PATCH - 1) / 2);
          var pixKernel = ee.Kernel.square({
            radius: radiusPixels,
            units: 'pixels',
            normalize: false
          });

          var neighborhoodArrays = composite.neighborhoodToArray(pixKernel);
          var patchArray = neighborhoodArrays.toArray().rename('patch');

          // total points and batches (client-side)
          allPoints.size().evaluate(function(totalPts){
            p('Total points to export:', totalPts);

            var numBatches = Math.ceil(totalPts / BATCH_SIZE);
            p('Will create', numBatches, 'export tasks with BATCH_SIZE=', BATCH_SIZE);

            for (var i = 0; i < numBatches; i++){
              (function(i){
                var low = i / numBatches;
                var high = (i + 1) / numBatches;

                var batchFC = allPoints.filter(
                  ee.Filter.and(
                    ee.Filter.gte('rand_final', low),
                    ee.Filter.lt('rand_final', high)
                  )
                );

                var trainingDataBatch = patchArray.sampleRegions({
                  collection: batchFC,
                  properties: ['label'],
                  scale: scale,
                  tileScale: TILE_SCALE,
                  geometries: true
                });

                var desc = 'Anantapur_batch_' + i + '_of_' + numBatches;

                Export.table.toDrive({
                  collection: trainingDataBatch,
                  description: desc,
                  folder: EXPORT_FOLDER,
                  fileFormat: 'TFRecord',
                  selectors: ['patch', 'label']
                });

                p('Started export task:', desc);
              })(i);
            }
          });
        });
      });
    });
  });
}