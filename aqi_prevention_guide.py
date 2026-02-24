"""
AQI Prevention and Health Recommendation Guide
Provides prevention steps and health recommendations based on AQI levels and predicted values
"""

from enum import Enum


class AQILevel(Enum):
    GOOD = (0, 50, 'Good')
    SATISFACTORY = (51, 100, 'Satisfactory')
    MODERATELY_POLLUTED = (101, 150, 'Moderately Polluted')
    POOR = (151, 200, 'Poor')
    VERY_POOR = (201, 300, 'Very Poor')
    SEVERE = (301, 9999, 'Severe')


class AQIPreventionGuide:
    """Provides health effects and prevention measures based on AQI levels"""
    
    # Health effects and warnings for each AQI level
    HEALTH_EFFECTS = {
        'Good': {
            'emoji': '🟢',
            'health_status': 'SAFE',
            'severity': 1,
            'color': '#28a745',
            'health_effects': [
                '✓ No adverse health effects',
                '✓ Excellent air quality for all activities',
                '✓ Safe for all population groups',
                '✓ Ideal for outdoor exercises and activities'
            ],
            'population_groups': {
                'general': 'All outdoor activities encouraged',
                'children': 'Safe for all outdoor activities',
                'elderly': 'Safe for all outdoor activities',
                'respiratory': 'Safe for all outdoor activities',
                'cardiac': 'Safe for all outdoor activities'
            }
        },
        'Satisfactory': {
            'emoji': '🟡',
            'health_status': 'SAFE',
            'severity': 2,
            'color': '#ffc107',
            'health_effects': [
                '⚠ Minor respiratory symptoms in sensitive groups',
                '⚠ Minimal risk for general population',
                '⚠ Occasional asthma symptoms in sensitive people',
                '⚠ Minor discomfort after prolonged outdoor exposure'
            ],
            'population_groups': {
                'general': 'Outdoor activities generally safe',
                'children': 'Can play outdoors but avoid intense activities',
                'elderly': 'Light outdoor activities safe; avoid strenuous activities',
                'respiratory': 'Use inhaler as needed; limit strenuous activities',
                'cardiac': 'Light activities safe'
            }
        },
        'Moderately Polluted': {
            'emoji': '🟠',
            'health_status': 'MODERATE HAZARD',
            'severity': 3,
            'color': '#fd7e14',
            'health_effects': [
                '⛔ Respiratory discomfort for sensitive groups',
                '⛔ Increased asthma attacks and coughing',
                '⛔ Difficulty in breathing for children & elderly',
                '⛔ General fatigue and reduced physical performance',
                '⛔ Eye and throat irritation possible'
            ],
            'population_groups': {
                'general': 'Reduce prolonged outdoor activities',
                'children': 'Avoid outdoor play; stay indoors as much as possible',
                'elderly': 'Reduce outdoor activities; stay in well-ventilated areas',
                'respiratory': 'Keep inhaler handy; minimize outdoor time',
                'cardiac': 'Avoid strenuous activities outdoors'
            }
        },
        'Poor': {
            'emoji': '🔴',
            'health_status': 'HAZARDOUS',
            'severity': 4,
            'color': '#dc3545',
            'health_effects': [
                '🚨 Severe respiratory illness in general population',
                '🚨 Increased heart disease risk',
                '🚨 Exacerbated asthma and cardiovascular diseases',
                '🚨 Reduced lung function and physical capacity',
                '🚨 Hospital admissions likely'
            ],
            'population_groups': {
                'general': 'STAY INDOORS - minimize outdoor exposure',
                'children': 'MUST stay indoors - no outdoor activities',
                'elderly': 'MUST stay indoors - critical health risk',
                'respiratory': 'Use N95 masks if must go outside; take medications',
                'cardiac': 'STAY INDOORS - avoid physical exertion'
            }
        },
        'Very Poor': {
            'emoji': '🟣',
            'health_status': 'VERY HAZARDOUS',
            'severity': 5,
            'color': '#6f42c1',
            'health_effects': [
                '🚨 Life-threatening respiratory illness',
                '🚨 Severe cardiovascular complications',
                '🚨 Hospital emergencies & mortality risk',
                '🚨 Mass respiratory symptoms expected',
                '🚨 Work capacity severely reduced'
            ],
            'population_groups': {
                'general': 'CRITICAL - Stay indoors with air filters ON',
                'children': 'CRITICAL - Must stay indoors; educational institutions closed',
                'elderly': 'CRITICAL - Stay indoors; emergency preparedness essential',
                'respiratory': 'Use N95/N100 masks; have medical help on standby',
                'cardiac': 'CRITICAL - Stay indoors; consult doctor for medications'
            }
        },
        'Severe': {
            'emoji': '💀',
            'health_status': 'EMERGENCY',
            'severity': 6,
            'color': '#721c24',
            'health_effects': [
                '💀 Life-threatening conditions for ALL',
                '💀 Respiratory failure and cardiac arrest risk',
                '💀 Mass casualties and mortality possible',
                '💀 Mass evacuation recommended',
                '💀 Total lockdown in effect'
            ],
            'population_groups': {
                'general': 'LOCKDOWN - Stay indoors; use advanced air filtration',
                'children': 'LOCKDOWN - Schools shut; stay indoors with activated air filters',
                'elderly': 'LOCKDOWN - Hospitals on alert; stay indoors',
                'respiratory': 'Use N100 masks; seek immediate medical help',
                'cardiac': 'LOCKDOWN - Seek hospital shelter; critical condition'
            }
        }
    }
    
    # Prevention and mitigation measures
    PREVENTION_MEASURES = {
        'Good': {
            'home': [
                '✓ Open windows for natural ventilation',
                '✓ No need for air purifiers',
                '✓ Regular household cleaning suitable'
            ],
            'outdoor': [
                '✓ All outdoor activities safe',
                '✓ Exercise and sports recommended',
                '✓ Picnics and outdoor events encouraged'
            ],
            'transport': [
                '✓ Prefer walking or cycling',
                '✓ Use public transport',
                '✓ Outdoor commute safe'
            ],
            'work': [
                '✓ Open office windows',
                '✓ No restrictions on work activities',
                '✓ Outdoor meetings safe'
            ]
        },
        'Satisfactory': {
            'home': [
                '✓ Use air purifiers in bedrooms',
                '⚠ Close windows during peak traffic hours',
                '✓ Regular cleaning important'
            ],
            'outdoor': [
                '⚠ Reduce vigorous outdoor activities',
                '⚠ Sensitive people should limit exposure',
                '✓ Light outdoor activities okay'
            ],
            'transport': [
                '✓ Keep car windows closed in traffic',
                '✓ Use recycled air in AC',
                '✓ Use masks if sensitive'
            ],
            'work': [
                '✓ Consider hybrid work for sensitive employees',
                '⚠ Limit outdoor work',
                '✓ Provide N95 masks at workplace'
            ]
        },
        'Moderately Polluted': {
            'home': [
                '⚠ Use air purifiers with HEPA filters',
                '⚠ Keep windows closed',
                '⚠ Seal window gaps and doors',
                '✓ Use wet cleaning instead of sweeping'
            ],
            'outdoor': [
                '⛔ Reduce outdoor exposure significantly',
                '⛔ NO strenuous outdoor activities',
                '⛔ Children should stay mostly indoors',
                '⚠ Use N95 masks if must go outside'
            ],
            'transport': [
                '✓ Use private vehicles instead of public transport',
                '✓ Use vehicle air conditioning on recirculate mode',
                '⚠ Wear N95 masks if using public transport'
            ],
            'work': [
                '☑ Schools should issue N95 masks',
                '☑ Outdoor work shifted to early morning/evening',
                '☑ Work from home encouraged',
                '☑ Provide air filtration at workplace'
            ]
        },
        'Poor': {
            'home': [
                '⛔ Use industrial-grade air purifiers',
                '⛔ Keep windows and doors sealed',
                '⛔ Turn off all external AC vents',
                '⛔ Ready emergency oxygen if elderly/cardiac patients'
            ],
            'outdoor': [
                '🚨 STAY INDOORS - No outdoor activities',
                '🚨 If must go outside, use N95/N99 masks',
                '🚨 Elderly and children MUST stay indoors',
                '🚨 Avoid parks, playgrounds, public gathering areas'
            ],
            'transport': [
                '✓ Use private vehicles with AC on recirculate',
                '✓ Minimize outdoor commute time',
                '⛔ Avoid public transport'
            ],
            'work': [
                '🚨 Work from home mandatory',
                '🚨 Schools and colleges closed',
                '🚨 N95 masks mandatory if must go outside',
                '🚨 Outdoor events cancelled'
            ]
        },
        'Very Poor': {
            'home': [
                '🚨 Multiple HEPA air purifiers running',
                '🚨 Windows completely sealed',
                '🚨 Emergency oxygen on standby',
                '🚨 Hospital emergency numbers saved',
                '🚨 Prepare emergency kit'
            ],
            'outdoor': [
                '💀 TOTAL LOCKDOWN - Stay indoors',
                '💀 N100 masks mandatory if unavoidable',
                '💀 Minimize all outdoor exposure',
                '💀 Avoid outdoor areas entirely'
            ],
            'transport': [
                '⛔ Avoid all unnecessary travel',
                '⛔ Use fastest available transport',
                '⛔ Keep N100 masks in vehicle'
            ],
            'work': [
                '🚨 Total lockdown in effect',
                '🚨 Schools closed indefinitely',
                '🚨 Work from home mandatory',
                '🚨 Only essential services operate',
                '🚨 Emergency services on high alert'
            ]
        },
        'Severe': {
            'home': [
                '💀 Multiple industrial air purifiers',
                '💀 Rooms sealed with tape',
                '💀 Advanced air filtration essential',
                '💀 Medical oxygen and facilities ready',
                '💀 Evacuation ready if required'
            ],
            'outdoor': [
                '💀 COMPLETE LOCKDOWN',
                '💀 NO outdoor activity whatsoever',
                '💀 Stay in sealed room with HEPA filtration',
                '💀 Medical emergency on alert'
            ],
            'transport': [
                '💀 No movement except emergencies',
                '💀 Ambulances on high alert',
                '💀 Emergency transport available'
            ],
            'work': [
                '💀 MASS LOCKDOWN IN EFFECT',
                '💀 All institutions closed',
                '💀 All non-essential movement banned',
                '💀 Hospitals operating in emergency mode',
                '💀 Mass evacuation protocols active'
            ]
        }
    }
    
    # Location-specific avoidance recommendations
    LOCATION_AVOIDANCE = {
        'delhi': {
            'high_risk_areas': [
                'Lodhi Road (industrial traffic)',
                'ITC Maurya area (traffic congestion)',
                'Anand Vihar (truck entry point)',
                'Sector 1 Noida (truck traffic)',
                'NH-8 areas (highway traffic)',
                'Badarpur (industrial area)',
                'Naraina (industrial zone)'
            ],
            'safer_areas': [
                'Lodhi Garden (parkland)',
                'Delhi Ridge area (forest)',
                'Mehrauli area (less dense)',
                'Greater Noida (peripheral)',
                'Areas with more green cover'
            ]
        },
        'mumbai': {
            'high_risk_areas': [
                'Eastern Express Highway (busy road)',
                'Western Express Highway (traffic)',
                'CST area (traffic congestion)',
                'Central areas (high density)',
                'Port areas (industrial)'
            ],
            'safer_areas': [
                'Marine Drive area (seafront)',
                'Sanjay Gandhi National Park',
                'Powai Lake area',
                'Borivali National Park',
                'Mangrove-rich areas'
            ]
        },
        'bangalore': {
            'high_risk_areas': [
                'IT corridor areas (heavy traffic)',
                'Outer ring road (highway)',
                'Central business district',
                'Industrial parks',
                'Flyover areas (stagnant pollution)'
            ],
            'safer_areas': [
                'Lalbagh area',
                'Cubbon Park',
                'Vidhana Soudha area',
                'Nandi Hills area',
                'Suburban green zones'
            ]
        }
    }
    
    @staticmethod
    def get_aqi_category(aqi_value):
        """Determine AQI category from numerical value"""
        if aqi_value <= 50:
            return 'Good'
        elif aqi_value <= 100:
            return 'Satisfactory'
        elif aqi_value <= 150:
            return 'Moderately Polluted'
        elif aqi_value <= 200:
            return 'Poor'
        elif aqi_value <= 300:
            return 'Very Poor'
        else:
            return 'Severe'
    
    @staticmethod
    def get_health_effects(aqi_level):
        """Get health effects for an AQI level"""
        return AQIPreventionGuide.HEALTH_EFFECTS.get(aqi_level, {})
    
    @staticmethod
    def get_population_guidance(aqi_level, population_type):
        """Get guidance for specific population group"""
        health_data = AQIPreventionGuide.HEALTH_EFFECTS.get(aqi_level, {})
        pop_groups = health_data.get('population_groups', {})
        return pop_groups.get(population_type, 'Use general precautions')
    
    @staticmethod
    def get_prevention_measures(aqi_level, category):
        """Get prevention measures for specific category"""
        measures = AQIPreventionGuide.PREVENTION_MEASURES.get(aqi_level, {})
        return measures.get(category, [])
    
    @staticmethod
    def get_all_prevention_measures(aqi_level):
        """Get all prevention measures for an AQI level"""
        return AQIPreventionGuide.PREVENTION_MEASURES.get(aqi_level, {})
    
    @staticmethod
    def get_location_recommendations(city):
        """Get location-specific avoidance and safety recommendations"""
        city_lower = city.lower()
        return AQIPreventionGuide.LOCATION_AVOIDANCE.get(city_lower, {
            'high_risk_areas': ['Central business areas', 'Heavy traffic areas', 'Industrial zones'],
            'safer_areas': ['Parks and gardens', 'Suburban areas', 'Areas with green cover']
        })
    
    @staticmethod
    def generate_aqi_forecast_report(predictions_df):
        """Generate health and prevention report based on predictions"""
        if predictions_df.empty:
            return {}
        
        aqi_values = predictions_df['predicted_aqi'].values
        avg_aqi = aqi_values.mean()
        max_aqi = aqi_values.max()
        
        category = AQIPreventionGuide.get_aqi_category(avg_aqi)
        health_effects = AQIPreventionGuide.get_health_effects(category)
        
        report = {
            'average_aqi': round(avg_aqi, 2),
            'max_aqi': round(max_aqi, 2),
            'category': category,
            'emoji': health_effects.get('emoji', ''),
            'health_status': health_effects.get('health_status', ''),
            'severity': health_effects.get('severity', 0),
            'health_effects': health_effects.get('health_effects', []),
            'population_guidance': health_effects.get('population_groups', {}),
            'prevention_measures': AQIPreventionGuide.PREVENTION_MEASURES.get(category, {}),
            'worst_location': predictions_df.nlargest(1, 'predicted_aqi').to_dict('records')[0] if not predictions_df.empty else {},
            'best_location': predictions_df.nsmallest(1, 'predicted_aqi').to_dict('records')[0] if not predictions_df.empty else {}
        }
        
        return report


if __name__ == "__main__":
    # Example usage
    guide = AQIPreventionGuide()
    
    # Test for different AQI levels
    for aqi in [25, 75, 125, 175, 250, 350]:
        category = guide.get_aqi_category(aqi)
        print(f"\nAQI: {aqi} → Category: {category}")
        print(f"Health Effects: {guide.get_health_effects(category)['health_effects'][:2]}")
